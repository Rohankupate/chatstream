import os
import re
import json
import uuid
import numpy as np
import sqlite3
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import PyPDF2
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
EMBEDDINGS_FOLDER = 'embeddings'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 25 * 1024 * 1024

for folder in [UPLOAD_FOLDER, DATA_FOLDER, EMBEDDINGS_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def init_db():
    conn = sqlite3.connect('pdf_storage.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_pdfs (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            page_count INTEGER NOT NULL,
            file_size INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

embedding_model = None
qa_pipeline = None
faiss_index = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def get_qa_pipeline():
    global qa_pipeline
    if qa_pipeline is None:
        print("Loading QA model...")
        model_name = "distilbert-base-cased-distilled-squad"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text, len(pdf_reader.pages)
    except Exception as e:
        return None, 0

def chunk_text(text, chunk_size=500, overlap=150):
    text = re.sub(r'\s+', ' ', text.strip())
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            words = current_chunk.split()
            overlap_words = words[-overlap//5:] if len(words) > overlap//5 else words[-10:]
            current_chunk = " ".join(overlap_words) + " " + paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.5:
            sentences = chunk.split('.')
            sub_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sub_chunk) + len(sentence) > chunk_size and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    words = sub_chunk.split()
                    overlap_words = words[-20:] if len(words) > 20 else words[-10:]
                    sub_chunk = " ".join(overlap_words) + ". " + sentence
                else:
                    sub_chunk += ". " + sentence if sub_chunk else sentence
            if sub_chunk.strip():
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)
    
    return final_chunks if final_chunks else [chunk for chunk in chunks if len(chunk.split()) > 20]

def create_embeddings(chunks, session_id):
    model = get_embedding_model()
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    index_path = os.path.join(EMBEDDINGS_FOLDER, f'{session_id}.index')
    chunks_path = os.path.join(EMBEDDINGS_FOLDER, f'{session_id}_chunks.json')
    
    faiss.write_index(index, index_path)
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)
    
    return index, chunks

def load_embeddings(session_id):
    try:
        index_path = os.path.join(EMBEDDINGS_FOLDER, f'{session_id}.index')
        chunks_path = os.path.join(EMBEDDINGS_FOLDER, f'{session_id}_chunks.json')
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            index = faiss.read_index(index_path)
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            return index, chunks
    except Exception as e:
        print(f"Error loading embeddings: {e}")
    
    return None, None

def retrieve_relevant_chunks(question, session_id, top_k=15):
    index, chunks = load_embeddings(session_id)
    
    if index is None or chunks is None:
        return []
    
    model = get_embedding_model()
    
    query_variations = [
        question,
        question.lower(),
        question.replace("what is", "").replace("define", "").replace("explain", "").strip(),
        question.replace("?", "").strip(),
        " ".join([word for word in question.split() if len(word) > 3 and word.lower() not in ['what', 'define', 'explain', 'describe']])
    ]
    
    all_relevant_chunks = []
    
    for query in query_variations:
        if not query.strip():
            continue
            
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding.astype('float32'), top_k)
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks) and score > 0.05:
                chunk_text = chunks[idx]
                if not any(existing['text'] == chunk_text for existing in all_relevant_chunks):
                    all_relevant_chunks.append({
                        'text': chunk_text,
                        'score': float(score),
                        'query': query
                    })
    
    all_relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
    return all_relevant_chunks[:12]

def extract_specific_answers(question, context):
    import re
    question_lower = question.lower()
    context_lower = context.lower()
    if any(word in question_lower for word in ['year', 'when', 'date']):
        year_patterns = [
            r'\b(19\d{2}|20\d{2})\b',  # Complete 4-digit years
            r'\b(\d{4})s\b',           # decades like 1980s
            r'in (\d{4})',             # "in 1989"
            r'(\d{4}) .*?(?:coined|began|started|originated)',  # year before action
            r'(?:coined|began|started|originated) .*?(\d{4})',  # action before year
            r'(\d{4})',                # any 4-digit number as fallback
        ]
        
        sentences = context.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains question keywords
            question_words = re.findall(r'\b\w+\b', question_lower)
            question_words = [word for word in question_words if len(word) > 2 and word not in ['when', 'what', 'year', 'date', 'did']]
            
            if any(word in sentence_lower for word in question_words):
                for pattern in year_patterns:
                    years = re.findall(pattern, sentence)
                    if years:
                        year = years[0]
                        # Validate it's a reasonable year
                        if year.isdigit() and 1900 <= int(year) <= 2030:
                            return f"{year}"
    
    # Name/person questions
    if any(word in question_lower for word in ['who', 'name', 'person', 'coined', 'created', 'invented']):
        sentences = context.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            question_words = re.findall(r'\b\w+\b', question_lower)
            question_words = [word for word in question_words if len(word) > 2 and word not in ['who', 'what', 'name', 'person']]
            
            if any(word in sentence_lower for word in question_words):
                # Look for proper names with more specific patterns
                name_patterns = [
                    r'([A-Z][a-z]+ [A-Z][a-z]+) (?:coined|created|originated|invented)',  # Name + action
                    r'(?:coined|created|originated|invented) (?:by )?([A-Z][a-z]+ [A-Z][a-z]+)',  # Action + name
                    r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Any proper name
                    r'\b([A-Z]\. [A-Z][a-z]+)\b',      # F. Last
                    r'\b([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)\b'  # First F. Last
                ]
                
                for pattern in name_patterns:
                    names = re.findall(pattern, sentence)
                    if names:
                        return names[0]
    
    return None

def generate_answer_with_qa_model(question, context):
    specific_answer = extract_specific_answers(question, context)
    if specific_answer:
        return specific_answer
    return enhanced_text_search(question, context)

def enhanced_text_search(question, context):
    question_lower = question.lower()
    
    # Define question types and their keywords
    question_types = {
        'definition': ['what is', 'define', 'definition', 'meaning', 'explain'],
        'lifecycle': ['lifecycle', 'process', 'stages', 'phases', 'steps'],
        'description': ['describe', 'explain', 'tell me about'],
        'comparison': ['difference', 'compare', 'versus', 'vs'],
        'list': ['types', 'kinds', 'examples', 'list']
    }
    
    # Identify question type
    question_type = 'general'
    for qtype, keywords in question_types.items():
        if any(keyword in question_lower for keyword in keywords):
            question_type = qtype
            break
    
    # Split context into sentences and paragraphs
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
    paragraphs = [p.strip() for p in context.split('\n') if len(p.strip()) > 50]
    
    # Extract key terms from question
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'how', 'why', 'when', 'where', 'who'}
    question_words = [word for word in question_lower.split() if len(word) > 2 and word not in stop_words]
    
    # Score and rank content
    scored_content = []
    
    # Score sentences
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0
        
        # Base keyword matching
        for word in question_words:
            if word in sentence_lower:
                score += 2
        
        # Bonus for question type specific patterns
        if question_type == 'definition' and any(pattern in sentence_lower for pattern in ['is a', 'refers to', 'defined as', 'means']):
            score += 5
        elif question_type == 'lifecycle' and any(pattern in sentence_lower for pattern in ['lifecycle', 'process', 'stages', 'phases']):
            score += 5
        
        if score > 0:
            scored_content.append((sentence, score, 'sentence'))
    
    # Score paragraphs for longer explanations
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        score = 0
        
        for word in question_words:
            score += paragraph_lower.count(word)
        
        if score > 2:  # Only include paragraphs with multiple matches
            scored_content.append((paragraph, score, 'paragraph'))
    
    # Sort by score
    scored_content.sort(key=lambda x: x[1], reverse=True)
    
    if not scored_content:
        return "I couldn't find relevant information to answer your question."
    
    # Return best match based on question type
    best_content = scored_content[0]
    
    if question_type in ['definition', 'lifecycle'] and best_content[2] == 'paragraph':
        # For definitions and explanations, prefer longer content
        answer = best_content[0]
        if len(answer) > 300:
            # Extract most relevant sentences from paragraph
            para_sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
            relevant_sentences = []
            for sent in para_sentences[:5]:  # Take first 5 sentences
                if any(word in sent.lower() for word in question_words):
                    relevant_sentences.append(sent)
            if relevant_sentences:
                return '. '.join(relevant_sentences[:3]) + '.'
        return answer
    else:
        # For other questions, return the best sentence
        return best_content[0]

def fallback_text_search(question, context):
    return enhanced_text_search(question, context)

def rag_search(question, session_id):
    print(f"\n=== RAG Search for: {question} ===")
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(question, session_id)
    
    if not relevant_chunks:
        return "I couldn't find relevant information in the uploaded PDF to answer your question."
    
    print(f"Found {len(relevant_chunks)} relevant chunks")
    
    # Combine context from top chunks with better organization
    context_parts = []
    for i, chunk in enumerate(relevant_chunks[:8]):
        context_parts.append(chunk['text'])
    
    context = "\n\n".join(context_parts)
    
    # Try specific answer extraction first
    specific_answer = extract_specific_answers(question, context)
    if specific_answer:
        return f"The answer is {specific_answer}"
    
    # Enhanced answer generation for definitions and explanations
    answer = generate_comprehensive_answer(question, context, relevant_chunks)
    
    return answer

def generate_comprehensive_answer(question, context, chunks):
    question_lower = question.lower()
    
    # Check if it's a definition question
    if any(word in question_lower for word in ['what is', 'define', 'definition', 'explain']):
        return generate_definition_answer(question, context, chunks)
    
    # Check if it's a comparison question
    if any(word in question_lower for word in ['difference', 'compare', 'versus', 'vs']):
        return generate_comparison_answer(question, context, chunks)
    
    # Default comprehensive answer
    return generate_general_answer(question, context, chunks)

def generate_definition_answer(question, context, chunks):
    concept = extract_main_concept(question)
    
    # Build a comprehensive answer by finding and organizing relevant information
    definition_parts = []
    explanatory_parts = []
    example_parts = []
    
    # Process all chunks to extract different types of information
    for chunk in chunks[:10]:
        chunk_text = chunk['text']
        
        # Split into sentences and clean them
        sentences = []
        for sent in chunk_text.replace('\n', ' ').split('.'):
            sent = sent.strip()
            if sent and len(sent.split()) > 5:
                sentences.append(sent)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            concept_words = concept.lower().split()
            
            # Check if sentence contains the main concept
            if any(word in sentence_lower for word in concept_words):
                
                # Categorize the sentence based on its content
                if any(pattern in sentence_lower for pattern in [
                    'is defined as', 'refers to', 'means', 'is a type of', 'is the', 'are methods'
                ]):
                    definition_parts.append(sentence)
                
                elif any(pattern in sentence_lower for pattern in [
                    'involves', 'consists of', 'includes', 'requires', 'uses', 'applies'
                ]):
                    explanatory_parts.append(sentence)
                
                elif any(pattern in sentence_lower for pattern in [
                    'example', 'for instance', 'such as', 'like', 'including'
                ]):
                    example_parts.append(sentence)
                
                elif len(sentence.split()) > 12 and any(word in sentence_lower for word in [
                    'learning', 'data', 'algorithm', 'method', 'technique', 'approach'
                ]):
                    explanatory_parts.append(sentence)
    
    # Build the comprehensive answer
    answer_parts = []
    
    # Start with definitions
    if definition_parts:
        answer_parts.extend(definition_parts[:2])
    
    # Add explanatory information
    if explanatory_parts:
        answer_parts.extend(explanatory_parts[:2])
    
    # Add examples if available
    if example_parts:
        answer_parts.extend(example_parts[:1])
    
    # If we have good content, format it properly
    if answer_parts:
        # Remove duplicates while preserving order
        unique_parts = []
        seen_content = set()
        for part in answer_parts:
            # Check for substantial overlap to avoid near-duplicates
            part_words = set(part.lower().split())
            is_duplicate = False
            for seen in seen_content:
                seen_words = set(seen.lower().split())
                overlap = len(part_words.intersection(seen_words))
                if overlap > len(part_words) * 0.7:  # 70% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_parts.append(part)
                seen_content.add(part)
        
        # Combine and format the answer
        result = '. '.join(unique_parts[:4])
        if result and not result.endswith('.'):
            result += '.'
        
        # Clean up formatting
        result = ' '.join(result.split())  # Remove extra whitespace
        return result
    
    # Fallback to simpler approach
    return generate_simple_definition(question, context, chunks)

def generate_simple_definition(question, context, chunks):
    concept = extract_main_concept(question)
    
    # Find the most relevant sentences
    scored_sentences = []
    
    for chunk in chunks[:8]:
        sentences = [s.strip() for s in chunk['text'].replace('\n', ' ').split('.') if s.strip()]
        
        for sentence in sentences:
            if len(sentence.split()) < 8:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            concept_lower = concept.lower()
            
            # Score sentences based on relevance
            score = 0
            
            # High score for direct concept mentions
            if concept_lower in sentence_lower:
                score += 10
            
            # Bonus for definition patterns
            if any(pattern in sentence_lower for pattern in [
                'is', 'are', 'refers to', 'means', 'involves', 'includes'
            ]):
                score += 5
            
            # Bonus for learning-related terms
            if any(term in sentence_lower for term in [
                'learning', 'data', 'training', 'algorithm', 'method'
            ]):
                score += 3
            
            if score > 8:  # Only include high-scoring sentences
                scored_sentences.append((score, sentence))
    
    # Sort by score and take the best ones
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    best_sentences = [sent for score, sent in scored_sentences[:3]]
    
    if best_sentences:
        result = '. '.join(best_sentences)
        if result and not result.endswith('.'):
            result += '.'
        return result
    
    # Final fallback
    return enhanced_text_search(question, context)

def extract_main_concept(question):
    """Extract the main concept from a question"""
    # Remove question words and common phrases
    cleaned = question.lower()
    for phrase in ['what is', 'define', 'explain', 'describe', 'what are']:
        cleaned = cleaned.replace(phrase, '')
    
    # Remove punctuation and get the main terms
    import string
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    words = cleaned.split()
    
    # Handle compound concepts like "supervised learning" or "unsupervised learning"
    if 'learning' in words:
        # Find learning-related concepts
        learning_concepts = []
        for i, word in enumerate(words):
            if word == 'learning' and i > 0:
                learning_concepts.append(f"{words[i-1]} learning")
        if learning_concepts:
            return learning_concepts[0]
    
    # Return the longest meaningful phrase (usually the concept)
    if len(words) >= 2:
        return ' '.join(words[:3])  # Take first few words as concept
    elif words:
        return words[0]
    return question

def clean_and_organize_text(text, concept):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    relevant_sentences = []
    for sentence in sentences:
        if concept.lower() in sentence.lower():
            if any(pattern in sentence.lower() for pattern in ['is defined as', 'refers to', 'means', 'is a', 'are']):
                relevant_sentences.insert(0, sentence)
            else:
                relevant_sentences.append(sentence)
    
    if not relevant_sentences:
        relevant_sentences = sentences[:3]
    
    result = '. '.join(relevant_sentences[:4])
    result = re.sub(r'\s+', ' ', result)
    result = result.strip()
    
    if result and not result.endswith('.'):
        result += '.'
    
    return result if result else "I found some information but couldn't extract a clear definition."

def generate_comparison_answer(question, context, chunks):
    return enhanced_text_search(question, context)

def generate_general_answer(question, context, chunks):
    return enhanced_text_search(question, context)

def save_pdf_data(session_id, text, filename, pages):
    data = {
        'text': text,
        'filename': filename,
        'pages': pages,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(DATA_FOLDER, f'{session_id}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_pdf_data(session_id):
    try:
        with open(os.path.join(DATA_FOLDER, f'{session_id}.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def delete_pdf_data(session_id):
    try:
        os.remove(os.path.join(DATA_FOLDER, f'{session_id}.json'))
        
        index_path = os.path.join(EMBEDDINGS_FOLDER, f'{session_id}.index')
        chunks_path = os.path.join(EMBEDDINGS_FOLDER, f'{session_id}_chunks.json')
        
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
    except FileNotFoundError:
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/get_uploaded_pdfs')
def get_uploaded_pdfs():
    """Get list of uploaded PDFs from database and restore session"""
    conn = sqlite3.connect('pdf_storage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM uploaded_pdfs ORDER BY upload_date DESC')
    pdfs = cursor.fetchall()
    conn.close()
    
    if pdfs:
        # Return the most recent PDF and set session
        pdf = pdfs[0]
        session['session_id'] = pdf[0]  # Set session ID to PDF ID
        
        # Verify that the data files exist for this session
        data_file = os.path.join(DATA_FOLDER, f"{pdf[0]}.json")
        embeddings_file = os.path.join(EMBEDDINGS_FOLDER, f"{pdf[0]}.index")
        
        if os.path.exists(data_file) and os.path.exists(embeddings_file):
            return jsonify({
                'success': True,
                'pdf': {
                    'id': pdf[0],
                    'filename': pdf[1],
                    'original_filename': pdf[2],
                    'upload_date': pdf[3],
                    'page_count': pdf[4],
                    'file_size': pdf[5]
                }
            })
        else:
            # Clean up database entry if files don't exist
            conn = sqlite3.connect('pdf_storage.db')
            cursor = conn.cursor()
            cursor.execute('DELETE FROM uploaded_pdfs WHERE id = ?', (pdf[0],))
            conn.commit()
            conn.close()
            return jsonify({'success': False})
    else:
        return jsonify({'success': False})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Check file size
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    
    if file and allowed_file(file.filename):
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Save file
        filename = secure_filename(file.filename)
        original_filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Extract text
        text, page_count = extract_text_from_pdf(file_path)
        if text is None:
            return jsonify({'error': 'Failed to extract text from PDF'}), 400
        
        # Save to database
        conn = sqlite3.connect('pdf_storage.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO uploaded_pdfs (id, filename, original_filename, upload_date, page_count, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, filename, original_filename, datetime.now().isoformat(), page_count, file_size))
        conn.commit()
        conn.close()
        
        # Save extracted text
        save_pdf_data(session_id, text, filename, page_count)
        
        # Generate embeddings
        chunks = chunk_text(text)
        create_embeddings(chunks, session_id)
        
        return jsonify({
            'success': True,
            'filename': original_filename,
            'pages': page_count,
            'session_id': session_id
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        if 'session_id' not in session:
            return jsonify({'error': 'Please upload a PDF first'}), 400
        
        # Load PDF data to verify it exists
        pdf_data = load_pdf_data(session['session_id'])
        if not pdf_data:
            return jsonify({'error': 'Please upload a PDF first'}), 400
        
        # Use RAG to get answer
        answer = rag_search(question, session['session_id'])
        
        return jsonify({
            'question': question,
            'answer': answer
        })
    except Exception as e:
        print(f"Error in ask_question: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    # Clear all PDFs from database (not just current session)
    conn = sqlite3.connect('pdf_storage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM uploaded_pdfs')
    all_pdfs = cursor.fetchall()
    
    # Clear all session data for all PDFs
    for pdf_row in all_pdfs:
        delete_pdf_data(pdf_row[0])
    
    # Remove all PDFs from database
    cursor.execute('DELETE FROM uploaded_pdfs')
    conn.commit()
    conn.close()
    
    # Clear current session
    session.clear()
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000 )
