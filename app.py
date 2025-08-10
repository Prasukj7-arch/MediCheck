# -----------------------------------------------------------------------------
# 1. IMPORTS
# -----------------------------------------------------------------------------

# --- Standard Library Imports ---
import os
import json
import logging
import io
import re
import base64
from functools import wraps

# --- Third-Party Imports ---
import PyPDF2
import numpy as np
import cv2
import openai
import chromadb
from chromadb.config import Settings
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from PIL import Image

# --- Local Application Imports ---
# Use a try-except block for optional modules like the symptom scanner
try:
    from backend.symptom_scanner import SymptomScanner
except ImportError:
    # Create a placeholder class if the module is not found
    SymptomScanner = None
    logging.warning("SymptomScanner module not found. Symptom scanning features will be disabled.")

# -----------------------------------------------------------------------------
# 2. INITIALIZATION AND CONFIGURATION
# -----------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a-very-strong-and-secret-key-for-development')

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- User Storage (Development Only) ---
# WARNING: This is an in-memory user store. It is NOT suitable for production.
# For a real application, use a proper database like PostgreSQL, MySQL, or a managed service.
users = {}

# -----------------------------------------------------------------------------
# 3. CORE RAG SYSTEM LOGIC
# -----------------------------------------------------------------------------

class MedicalRAGSystem:
    """A Retrieval-Augmented Generation system for medical document analysis."""
    
    def __init__(self):
        logger.info("Initializing MedicalRAGSystem...")
        
        # Initialize sentence transformer model for creating embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB persistent client (telemetry disabled for privacy/compatibility)
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create the collection for medical reports
        self.collection = self.chroma_client.get_or_create_collection(name="medical_reports")
        
        # Initialize OpenAI client for OpenRouter
        self.openai_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        
        # Make the LLM model configurable via environment variables
        self.llm_model = os.getenv('OPENROUTER_LLM_MODEL', 'mistralai/mistral-7b-instruct')
        logger.info(f"Using LLM model: {self.llm_model}")

    def extract_text_from_pdf(self, pdf_file_stream):
        """Extracts text from a PDF file stream."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
            text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Splits text into overlapping chunks, trying to respect sentence boundaries."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # If the chunk doesn't end at the end of the text, try to find a better break point
            if end < len(text):
                # Find the last sentence boundary ('.', '!', '?', or newline) in the chunk
                break_points = [chunk.rfind(p) for p in ['.', '!', '?', '\n']]
                last_break = max(break_points)
                
                # Use the boundary if it's within a reasonable part of the chunk (e.g., last 30%)
                # This avoids creating very small chunks.
                if last_break > chunk_size * 0.7:
                    chunk = chunk[:last_break + 1]
                    end = start + last_break + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            if start >= len(text):
                break
        
        return [c for c in chunks if c] # Filter out any empty chunks

    def create_embeddings(self, texts):
        """Creates vector embeddings for a list of text chunks."""
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def store_documents(self, chunks, embeddings, metadata_list):
        """Stores document chunks and their embeddings in ChromaDB."""
        if not chunks:
            logger.warning("No chunks to store.")
            return []
        try:
            # Generate unique IDs for each document chunk
            ids = [f"doc_{session.get('user_id', 'unknown')}_{i}_{os.urandom(4).hex()}" for i in range(len(chunks))]
            self.collection.add(embeddings=embeddings, documents=chunks, metadatas=metadata_list, ids=ids)
            logger.info(f"Stored {len(chunks)} document chunks in the vector database.")
            return ids
        except Exception as e:
            logger.error(f"Error storing documents in ChromaDB: {e}")
            raise

    def search_similar_chunks(self, query, top_k=5):
        """Searches for text chunks semantically similar to the query."""
        try:
            query_embedding = self.embedding_model.encode([query])
            results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
            return results
        except Exception as e:
            logger.error(f"Error searching for chunks: {e}")
            raise

    def is_medical_query(self, query):
        """
        Checks if a query is likely medical-related using keywords and patterns.
        This is a heuristic approach and may not be perfect.
        """
        medical_keywords = [
            'medical', 'health', 'doctor', 'patient', 'diagnosis', 'treatment', 'symptoms',
            'medication', 'medicine', 'prescription', 'test', 'lab', 'blood', 'urine',
            'x-ray', 'scan', 'mri', 'ct', 'ultrasound', 'biopsy', 'surgery', 'procedure',
            'condition', 'disease', 'illness', 'infection', 'pain', 'fever', 'cough',
            'headache', 'nausea', 'vomiting', 'diarrhea', 'constipation', 'fatigue',
            'weakness', 'dizziness', 'chest pain', 'shortness of breath', 'swelling',
            'rash', 'report', 'results', 'findings', 'normal', 'abnormal', 'elevated',
            'decreased', 'positive', 'negative', 'high', 'low', 'range', 'level',
            'count', 'pressure', 'cholesterol', 'glucose', 'hemoglobin', 'heart rate'
        ]
        
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in medical_keywords):
            return True

        # Build regex pattern from keywords to avoid repetition
        keyword_pattern = '|'.join(medical_keywords)
        medical_patterns = [
            r'\b(what|how|why|when|where|explain|describe)\b.*?\b(' + keyword_pattern + r')\b',
            r'\b(is my|are my)\b.*?\b(' + keyword_pattern + r')\b'
        ]

        if any(re.search(pattern, query_lower) for pattern in medical_patterns):
            return True
        
        return False

    def generate_response(self, query, context_chunks, perspective='patient'):
        """Generates a response using an LLM, tailored to the user's perspective."""
        try:
            # Guard clause: Ensure the query is medical-related
            if not self.is_medical_query(query):
                response_text = "I can only answer questions related to medical reports and health information. Please ask about your medical report, test results, symptoms, or other health topics."
                if perspective != 'patient':
                    response_text = "I can only provide clinical analysis for medical-related queries. Please ask about clinical findings, diagnoses, or treatments."
                return response_text

            context = "\n\n".join(context_chunks)
            
            if perspective == 'patient':
                system_message = "You are a friendly and empathetic medical assistant. Your role is to explain health information from medical reports to patients in simple, clear, and conversational language. Avoid jargon and be reassuring."
                prompt = f"""A patient has a question about their medical report. Using the provided context from their report, answer their question.
                
                **Medical Report Context:**
                ---
                {context}
                ---
                
                **Patient's Question:** "{query}"
                
                **Instructions:**
                1. Answer the question directly based on the context.
                2. Explain any medical terms in plain English.
                3. Keep your tone conversational, reassuring, and easy to understand.
                4. Conclude by advising the patient to always discuss their results with their doctor for a formal diagnosis and treatment plan.
                
                **Response:**"""
            else: # doctor perspective
                system_message = "You are an expert medical analyst. Your role is to provide a concise, professional summary of clinical findings for other healthcare professionals based on the provided report context."
                prompt = f"""Analyze the following medical report context to answer a clinical query.
                
                **Medical Report Context:**
                ---
                {context}
                ---
                
                **Clinical Query:** "{query}"
                
                **Instructions:**
                1. Provide a direct and professional analysis based on the context.
                2. Use appropriate medical terminology.
                3. Highlight key findings, their significance, and potential implications.
                4. If applicable, mention differential diagnoses or suggest next steps for investigation.
                
                **Analysis:**"""
            
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

# -----------------------------------------------------------------------------
# 4. GLOBAL INSTANTIATIONS
# -----------------------------------------------------------------------------

# Instantiate the core systems
rag_system = MedicalRAGSystem()

# Initialize the symptom scanner if the module is available and model file exists
symptom_scanner = None
if SymptomScanner:
    model_paths = ['models/symptom_model.pt', 'backend/models/symptom_model.pt', 'symptom_model.pt']
    model_path_found = next((path for path in model_paths if os.path.exists(path)), None)
    if model_path_found:
        try:
            symptom_scanner = SymptomScanner(model_path_found)
            logger.info("Symptom scanner initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SymptomScanner with model at {model_path_found}: {e}")
    else:
        logger.warning("Symptom model file not found in expected locations. Symptom scanner is disabled.")

# -----------------------------------------------------------------------------
# 5. FLASK DECORATORS & ROUTES
# -----------------------------------------------------------------------------

def login_required(f):
    """Decorator to ensure a user is logged in before accessing a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Authentication and Basic Pages ---

@app.route('/')
def index():
    return redirect(url_for('login') if 'user_id' not in session else url_for('homepage'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        user = users.get(email)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = email
            session['user_name'] = user['name']
            return jsonify({'success': True, 'redirect': url_for('homepage')})
        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        if not all([name, email, password]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        if email in users:
            return jsonify({'success': False, 'error': 'Email already registered'}), 409
        
        users[email] = {'name': name, 'password': generate_password_hash(password)}
        session['user_id'] = email
        session['user_name'] = name
        return jsonify({'success': True, 'redirect': url_for('homepage')})
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/homepage')
@login_required
def homepage():
    return render_template('homepage.html', user_name=session.get('user_name'))

@app.route('/report-analysis')
@login_required
def report_analysis():
    return render_template('report_analysis.html')

# --- Core API Endpoints ---

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handles PDF file upload, text extraction, and embedding."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        
        if file and allowed_file(file.filename):
            pdf_content = file.read()
            text = rag_system.extract_text_from_pdf(io.BytesIO(pdf_content))
            if not text.strip():
                return jsonify({'error': 'Could not extract any text from the PDF. The file might be image-based or empty.'}), 400
            
            chunks = rag_system.chunk_text(text)
            embeddings = rag_system.create_embeddings(chunks)
            metadata_list = [{"source": file.filename, "chunk_index": i} for i, chunk in enumerate(chunks)]
            rag_system.store_documents(chunks, embeddings, metadata_list)
            
            return jsonify({'message': f"Successfully processed '{file.filename}'", 'chunks_created': len(chunks)})
        else:
            return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        return jsonify({'error': 'An unexpected error occurred during file processing.'}), 500

@app.route('/query', methods=['POST'])
@login_required
def query():
    """Handles user queries against the uploaded documents."""
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        perspective = data.get('perspective', 'patient')
        
        if not query_text:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        search_results = rag_system.search_similar_chunks(query_text, top_k=5)
        context_chunks = search_results['documents'][0] if search_results and search_results['documents'] else []

        if not context_chunks:
            # Fallback response if no relevant context is found
            response_text = "I couldn't find any specific information about that in your uploaded documents. Could you try rephrasing your question or asking about something else?"
            return jsonify({'response': response_text})

        response = rag_system.generate_response(query_text, context_chunks, perspective)
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': 'An error occurred while processing your query.'}), 500

@app.route('/status')
def status():
    """Provides a health check and status of the system."""
    try:
        return jsonify({
            'status': 'healthy',
            'documents_in_db': rag_system.collection.count(),
            'embedding_model': 'all-MiniLM-L6-v2',
            'vector_db': 'ChromaDB',
            'llm_model_in_use': rag_system.llm_model,
            'symptom_scanner_status': 'available' if symptom_scanner else 'not_available'
        })
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# --- Symptom Scanner Endpoints ---

@app.route('/symptom-scanner/upload', methods=['POST'])
@login_required
def upload_symptom_image():
    """Uploads and scans an image for dermatological symptoms."""
    if not symptom_scanner:
        return jsonify({'error': 'The symptom scanner feature is not available.'}), 503
        
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        
        # Further checks...
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        
        scan_result = symptom_scanner.scan_image(filepath)
        condition_info = symptom_scanner.get_condition_info(scan_result['primary_condition'])
        
        return jsonify({'scan_result': scan_result, 'condition_info': condition_info})
    except Exception as e:
        logger.error(f"Error processing symptom image: {e}")
        return jsonify({'error': 'Failed to process image.'}), 500


@app.route('/symptom-scanner/scan-live', methods=['POST'])
@login_required
def scan_live_image():
    """Scans a live image from a base64 data URL."""
    if not symptom_scanner:
        return jsonify({'error': 'The symptom scanner feature is not available.'}), 503
        
    try:
        data = request.get_json()
        image_data = data['image_data'].split(',')[1] # Remove data URL prefix
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data received'}), 400
        
        scan_result = symptom_scanner.scan_cv2_image(image)
        condition_info = symptom_scanner.get_condition_info(scan_result['primary_condition'])

        return jsonify({'scan_result': scan_result, 'condition_info': condition_info})
    except Exception as e:
        logger.error(f"Error processing live image scan: {e}")
        return jsonify({'error': 'Failed to process live image.'}), 500

# -----------------------------------------------------------------------------
# 6. APP EXECUTION
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Use debug=False in a production environment
    app.run(debug=True, host='0.0.0.0', port=5000)