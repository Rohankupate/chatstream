import os
import re
import json
import uuid
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import PyPDF2
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text, len(pdf_reader.pages)
    except Exception as e:
        return None, 0

def save_pdf_data(session_id, text, filename, pages):
    """Save PDF data to file"""
    data = {
        'text': text,
        'filename': filename,
        'pages': pages,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(DATA_FOLDER, f'{session_id}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_pdf_data(session_id):
    """Load PDF data from file"""
    try:
        with open(os.path.join(DATA_FOLDER, f'{session_id}.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def delete_pdf_data(session_id):
    """Delete PDF data file"""
    try:
        os.remove(os.path.join(DATA_FOLDER, f'{session_id}.json'))
    except FileNotFoundError:
        pass

def extract_precise_answer(text, question):
    """Extract precise answers for specific question types"""
    question_lower = question.lower()
    
    # Year-based questions
    if any(word in question_lower for word in ['year', 'when', 'date']):
        # Look for years (1900-2099)
        year_pattern = r'\b(19|20)\d{2}\b'
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains question keywords
            question_words = re.findall(r'\b\w+\b', question_lower)
            question_words = [word for word in question_words if len(word) > 2 and word not in ['when', 'what', 'year', 'date']]
            
            if any(word in sentence_lower for word in question_words):
                years = re.findall(year_pattern, sentence)
                if years:
                    year = years[0] + re.findall(r'\b' + years[0] + r'(\d{2})\b', sentence)[0]
                    return f"The answer is {year}."
    
    # Name/person questions
    if any(word in question_lower for word in ['who', 'name', 'person', 'coined', 'created', 'invented']):
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            question_words = re.findall(r'\b\w+\b', question_lower)
            question_words = [word for word in question_words if len(word) > 2 and word not in ['who', 'what', 'name', 'person']]
            
            if any(word in sentence_lower for word in question_words):
                # Look for proper names (capitalized words)
                name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
                names = re.findall(name_pattern, sentence)
                if names:
                    return f"The answer is {names[0]}."
    
    return None

def simple_search(text, question):
    """Enhanced text search with precise answer extraction"""
    if not text or not question:
        return "Please upload a PDF and ask a question."
    
    # First try to extract a precise answer
    precise_answer = extract_precise_answer(text, question)
    if precise_answer:
        return precise_answer
    
    # Fall back to context-based search
    text_lower = text.lower()
    question_lower = question.lower()
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Keywords from the question (excluding common words)
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who'}
    question_words = re.findall(r'\b\w+\b', question_lower)
    question_words = [word for word in question_words if len(word) > 2 and word not in stop_words]
    
    # Find sentences that contain question keywords
    relevant_sentences = []
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if len(sentence_clean) < 20:  # Skip very short sentences
            continue
            
        sentence_lower = sentence_clean.lower()
        score = 0
        
        # Calculate relevance score
        for word in question_words:
            if word in sentence_lower:
                # Higher score for exact matches
                score += sentence_lower.count(word) * 2
        
        # Bonus for sentences with multiple keywords
        keyword_count = sum(1 for word in question_words if word in sentence_lower)
        if keyword_count > 1:
            score += keyword_count * 3
        
        if score > 0:
            relevant_sentences.append((sentence_clean, score))
    
    # Sort by relevance
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    
    if relevant_sentences:
        # Return the most relevant sentence, but limit length
        best_sentence = relevant_sentences[0][0]
        
        # If sentence is too long, try to extract the most relevant part
        if len(best_sentence) > 200:
            words = best_sentence.split()
            # Find the part with most question keywords
            best_part = ""
            max_keyword_density = 0
            
            for i in range(len(words) - 20):
                chunk = " ".join(words[i:i+20])
                chunk_lower = chunk.lower()
                keyword_count = sum(1 for word in question_words if word in chunk_lower)
                
                if keyword_count > max_keyword_density:
                    max_keyword_density = keyword_count
                    best_part = chunk
            
            return best_part if best_part else best_sentence[:200] + "..."
        
        return best_sentence
    else:
        return "I couldn't find information related to your question in the PDF."

@app.route('/')
def index():
    return render_template('index.html')

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
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': 'File size exceeds 10MB limit'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Extract text from PDF
        text, page_count = extract_text_from_pdf(file_path)
        
        if text is None:
            return jsonify({'error': 'Failed to extract text from PDF'}), 400
        
        # Generate session ID and store data in file
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        save_pdf_data(session['session_id'], text, filename, page_count)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'pages': page_count,
            'message': 'PDF uploaded successfully'
        })
    
    return jsonify({'error': 'Upload failed'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Please enter a question'}), 400
    
    if 'session_id' not in session:
        return jsonify({'error': 'Please upload a PDF first'}), 400
    
    # Load PDF data from file
    pdf_data = load_pdf_data(session['session_id'])
    if not pdf_data:
        return jsonify({'error': 'Please upload a PDF first'}), 400
    
    # Get answer using simple text search
    answer = simple_search(pdf_data['text'], question)
    
    return jsonify({
        'question': question,
        'answer': answer
    })

@app.route('/clear', methods=['POST'])
def clear_session():
    if 'session_id' in session:
        delete_pdf_data(session['session_id'])
    session.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
