import os
import json
import logging
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import PyPDF2
import io
import re
from sentence_transformers import SentenceTransformer
import chromadb
import openai
from dotenv import load_dotenv
import numpy as np
import requests
from functools import wraps
import cv2
from PIL import Image
import base64

# Import the symptom scanner
try:
    from backend.symptom_scanner import SymptomScanner
except ImportError:
    # If the module doesn't exist, we'll create a placeholder
    SymptomScanner = None

# Import the medical chatbot
try:
    from backend.medical_chatbot import MedicalChatbot
except ImportError:
    # If the module doesn't exist, we'll create a placeholder
    MedicalChatbot = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple user storage (in production, use a proper database)
users = {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize components
class MedicalRAGSystem:
    def __init__(self):
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB with new client configuration
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize OpenAI client for OpenRouter
        self.openai_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="medical_reports"
        )
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Find the last sentence boundary in the chunk
                last_period = chunk.rfind('.')
                last_exclamation = chunk.rfind('!')
                last_question = chunk.rfind('?')
                last_newline = chunk.rfind('\n')
                
                break_point = max(last_period, last_exclamation, last_question, last_newline)
                if break_point > chunk_size * 0.7:  # Only break if we're not too early
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def create_embeddings(self, texts):
        """Create embeddings for text chunks"""
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def store_documents(self, chunks, embeddings, metadata_list):
        """Store documents in ChromaDB"""
        try:
            # Generate IDs for documents
            ids = [f"doc_{i}" for i in range(len(chunks))]
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata_list,
                ids=ids
            )
            
            logger.info(f"Stored {len(chunks)} documents in vector database")
            return ids
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise
    
    def search_similar_chunks(self, query, top_k=5):
        """Search for similar chunks based on query"""
        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            raise
    
    def is_medical_query(self, query):
        """Check if the query is medical-related"""
        # Medical keywords and phrases
        medical_keywords = [
            'medical', 'health', 'doctor', 'patient', 'diagnosis', 'treatment', 'symptoms',
            'medication', 'medicine', 'prescription', 'test', 'lab', 'blood', 'urine',
            'x-ray', 'scan', 'mri', 'ct', 'ultrasound', 'biopsy', 'surgery', 'procedure',
            'condition', 'disease', 'illness', 'infection', 'pain', 'fever', 'cough',
            'headache', 'nausea', 'vomiting', 'diarrhea', 'constipation', 'fatigue',
            'weakness', 'dizziness', 'chest pain', 'shortness of breath', 'swelling',
            'rash', 'bruise', 'wound', 'injury', 'fracture', 'sprain', 'strain',
            'chronic', 'acute', 'allergy', 'asthma', 'diabetes', 'hypertension',
            'heart', 'lung', 'liver', 'kidney', 'brain', 'cancer', 'tumor',
            'report', 'results', 'findings', 'normal', 'abnormal', 'elevated',
            'decreased', 'positive', 'negative', 'high', 'low', 'range', 'level',
            'count', 'pressure', 'temperature', 'pulse', 'heart rate', 'blood pressure',
            'weight', 'height', 'bmi', 'cholesterol', 'glucose', 'hemoglobin',
            'white blood cell', 'red blood cell', 'platelet', 'protein', 'albumin',
            'bilirubin', 'creatinine', 'bun', 'sodium', 'potassium', 'chloride',
            'calcium', 'magnesium', 'phosphate', 'vitamin', 'mineral', 'hormone',
            'thyroid', 'adrenal', 'pituitary', 'pancreas', 'gallbladder', 'spleen',
            'lymph node', 'immune', 'antibody', 'antigen', 'vaccine', 'immunization',
            'pregnancy', 'obstetric', 'gynecologic', 'menstrual', 'fertility',
            'pediatric', 'geriatric', 'psychiatric', 'mental', 'depression', 'anxiety',
            'stress', 'sleep', 'appetite', 'digestion', 'metabolism', 'hormone',
            'endocrine', 'neurological', 'neurological', 'spinal', 'nervous',
            'musculoskeletal', 'joint', 'bone', 'muscle', 'tendon', 'ligament',
            'skin', 'dermatological', 'dental', 'oral', 'ophthalmic', 'eye',
            'ear', 'nose', 'throat', 'respiratory', 'cardiovascular', 'gastrointestinal',
            'genitourinary', 'reproductive', 'oncology', 'radiology', 'pathology',
            'pharmacy', 'pharmacology', 'therapeutic', 'dosage', 'side effect',
            'interaction', 'contraindication', 'precaution', 'monitoring', 'follow-up',
            'prognosis', 'outcome', 'recovery', 'rehabilitation', 'therapy', 'counseling'
        ]
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check if query contains medical keywords
        for keyword in medical_keywords:
            if keyword in query_lower:
                return True
        
        # Check for medical question patterns
        medical_patterns = [
            r'\b(what|how|why|when|where)\b.*\b(medical|health|doctor|patient|diagnosis|treatment|symptoms|medication|test|lab|blood|urine|x-ray|scan|mri|ct|ultrasound|biopsy|surgery|procedure|condition|disease|illness|infection|pain|fever|cough|headache|nausea|vomiting|diarrhea|constipation|fatigue|weakness|dizziness|chest pain|shortness of breath|swelling|rash|bruise|wound|injury|fracture|sprain|strain|chronic|acute|allergy|asthma|diabetes|hypertension|heart|lung|liver|kidney|brain|cancer|tumor|report|results|findings|normal|abnormal|elevated|decreased|positive|negative|high|low|range|level|count|pressure|temperature|pulse|heart rate|blood pressure|weight|height|bmi|cholesterol|glucose|hemoglobin|white blood cell|red blood cell|platelet|protein|albumin|bilirubin|creatinine|bun|sodium|potassium|chloride|calcium|magnesium|phosphate|vitamin|mineral|hormone|thyroid|adrenal|pituitary|pancreas|gallbladder|spleen|lymph node|immune|antibody|antigen|vaccine|immunization|pregnancy|obstetric|gynecologic|menstrual|fertility|pediatric|geriatric|psychiatric|mental|depression|anxiety|stress|sleep|appetite|digestion|metabolism|hormone|endocrine|neurological|neurological|spinal|nervous|musculoskeletal|joint|bone|muscle|tendon|ligament|skin|dermatological|dental|oral|ophthalmic|eye|ear|nose|throat|respiratory|cardiovascular|gastrointestinal|genitourinary|reproductive|oncology|radiology|pathology|pharmacy|pharmacology|therapeutic|dosage|side effect|interaction|contraindication|precaution|monitoring|follow-up|prognosis|outcome|recovery|rehabilitation|therapy|counseling)\b',
            r'\b(explain|describe|tell me about|what does|what is|what are|how do|how does|why do|why does|when do|when does|where do|where does)\b.*\b(medical|health|doctor|patient|diagnosis|treatment|symptoms|medication|test|lab|blood|urine|x-ray|scan|mri|ct|ultrasound|biopsy|surgery|procedure|condition|disease|illness|infection|pain|fever|cough|headache|nausea|vomiting|diarrhea|constipation|fatigue|weakness|dizziness|chest pain|shortness of breath|swelling|rash|bruise|wound|injury|fracture|sprain|strain|chronic|acute|allergy|asthma|diabetes|hypertension|heart|lung|liver|kidney|brain|cancer|tumor|report|results|findings|normal|abnormal|elevated|decreased|positive|negative|high|low|range|level|count|pressure|temperature|pulse|heart rate|blood pressure|weight|height|bmi|cholesterol|glucose|hemoglobin|white blood cell|red blood cell|platelet|protein|albumin|bilirubin|creatinine|bun|sodium|potassium|chloride|calcium|magnesium|phosphate|vitamin|mineral|hormone|thyroid|adrenal|pituitary|pancreas|gallbladder|spleen|lymph node|immune|antibody|antigen|vaccine|immunization|pregnancy|obstetric|gynecologic|menstrual|fertility|pediatric|geriatric|psychiatric|mental|depression|anxiety|stress|sleep|appetite|digestion|metabolism|hormone|endocrine|neurological|neurological|spinal|nervous|musculoskeletal|joint|bone|muscle|tendon|ligament|skin|dermatological|dental|oral|ophthalmic|eye|ear|nose|throat|respiratory|cardiovascular|gastrointestinal|genitourinary|reproductive|oncology|radiology|pathology|pharmacy|pharmacology|therapeutic|dosage|side effect|interaction|contraindication|precaution|monitoring|follow-up|prognosis|outcome|recovery|rehabilitation|therapy|counseling)\b'
        ]
        
        import re
        for pattern in medical_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False

    def generate_response(self, query, context_chunks, perspective='patient'):
        """Generate response using LLM with perspective-specific prompts"""
        try:
            # First check if the query is medical-related
            if not self.is_medical_query(query):
                if perspective == 'patient':
                    return "I can only answer questions related to medical reports and health information. Please ask me about your medical report, test results, medications, symptoms, or other health-related topics."
                else:
                    return "I can only provide clinical analysis for medical-related queries. Please ask me about medical reports, clinical findings, diagnoses, treatments, or other healthcare-related topics."
            
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            if perspective == 'patient':
                # Patient-friendly prompt - normal conversational style
                prompt = f"""You are a helpful medical assistant helping a patient understand their medical report. 
                
                Medical Report Context:
                {context}
                
                Patient Question: {query}
                
                Please provide a clear, conversational response that:
                1. Uses simple, everyday language
                2. Explains medical terms in plain English
                3. Is helpful and informative
                4. Gives practical advice when relevant
                5. Encourages them to ask their doctor if they have concerns
                
                Write in a normal, conversational tone - not like a formal letter. Be direct and helpful.
                
                Response:"""
                
                system_message = "You are a helpful medical assistant who explains health information in simple, conversational terms."
                
            else:
                # Doctor/clinical prompt - professional but not overly formal
                prompt = f"""You are a medical expert analyzing a medical report for clinical decision-making. 
                
                Medical Report Context:
                {context}
                
                Clinical Question: {query}
                
                Please provide a detailed, professional response that includes:
                1. Key clinical findings and their significance
                2. Relevant medical terminology and pathophysiology
                3. Differential diagnosis considerations
                4. Evidence-based treatment recommendations
                5. Follow-up and monitoring recommendations
                6. Any red flags or concerning findings
                
                Use appropriate medical terminology but keep the tone professional and clear, not overly formal.
                
                Response:"""
                
                system_message = "You are a medical expert providing professional analysis and recommendations in a clear, accessible manner."
            
            # Call OpenRouter API
            response = self.openai_client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

# Initialize RAG system
rag_system = MedicalRAGSystem()

# Initialize symptom scanner
symptom_scanner = None
if SymptomScanner:
    try:
        # Look for model file in common locations
        model_paths = [
            'models/symptom_model.pt',
            'backend/models/symptom_model.pt',
            'symptom_model.pt'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        # Initialize OpenAI client for LLM integration
        openai_client = None
        if os.getenv('OPENROUTER_API_KEY'):
            openai_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv('OPENROUTER_API_KEY')
            )
            logger.info("OpenAI client initialized for LLM integration")
        else:
            logger.warning("OPENROUTER_API_KEY not found. LLM features will be limited.")
        
        symptom_scanner = SymptomScanner(model_path, openai_client)
        logger.info("Symptom scanner initialized successfully with LLM integration")
    except Exception as e:
        logger.error(f"Error initializing symptom scanner: {e}")
        symptom_scanner = None
else:
    logger.warning("SymptomScanner module not available")

# Initialize medical chatbot
medical_chatbot = None
if MedicalChatbot:
    try:
        # Initialize OpenAI client for LLM integration
        openai_client = None
        if os.getenv('OPENROUTER_API_KEY'):
            openai_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv('OPENROUTER_API_KEY')
            )
            logger.info("OpenAI client initialized for Medical Chatbot")
        else:
            logger.warning("OPENROUTER_API_KEY not found. Medical Chatbot features will be limited.")
        
        medical_chatbot = MedicalChatbot(openai_client)
        logger.info("Medical Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing medical chatbot: {e}")
        medical_chatbot = None
else:
    logger.warning("MedicalChatbot module not available")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    if 'user_id' in session:
        return redirect(url_for('homepage'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login"""
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if email in users and check_password_hash(users[email]['password'], password):
            session['user_id'] = email
            session['user_name'] = users[email]['name']
            return jsonify({'success': True, 'redirect': url_for('homepage')})
        else:
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle signup"""
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if email in users:
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        
        users[email] = {
            'name': name,
            'password': generate_password_hash(password)
        }
        
        session['user_id'] = email
        session['user_name'] = name
        return jsonify({'success': True, 'redirect': url_for('homepage')})
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    """Handle logout"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/homepage')
@login_required
def homepage():
    """Serve the homepage with sidebar"""
    return render_template('homepage.html', user_name=session.get('user_name'))

@app.route('/report-analysis')
@login_required
def report_analysis():
    """Serve the report analysis page"""
    return render_template('report_analysis.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle PDF file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read PDF content
            pdf_content = file.read()
            
            # Extract text from PDF
            text = rag_system.extract_text_from_pdf(io.BytesIO(pdf_content))
            
            if not text.strip():
                return jsonify({'error': 'No text could be extracted from the PDF'}), 400
            
            # Chunk the text
            chunks = rag_system.chunk_text(text)
            
            # Create embeddings
            embeddings = rag_system.create_embeddings(chunks)
            
            # Prepare metadata
            metadata_list = [
                {
                    "source": file.filename,
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Store in vector database
            rag_system.store_documents(chunks, embeddings, metadata_list)
            
            return jsonify({
                'message': 'PDF processed successfully',
                'chunks_created': len(chunks),
                'filename': file.filename
            })
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400
    
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
@login_required
def query():
    """Handle user queries with perspective-specific responses"""
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        perspective = data.get('perspective', 'patient')  # Default to patient view
        
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
        
        # Search for relevant chunks
        search_results = rag_system.search_similar_chunks(query_text, top_k=5)
        
        if not search_results['documents'] or not search_results['documents'][0]:
            return jsonify({'error': 'No relevant information found in the uploaded documents'}), 404
        
        # Get the most relevant chunks
        context_chunks = search_results['documents'][0]
        
        # Generate response with perspective-specific prompt
        response = rag_system.generate_response(query_text, context_chunks, perspective)
        
        return jsonify({
            'response': response,
            'query': query_text,
            'perspective': perspective,
            'chunks_used': len(context_chunks)
        })
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot():
    """Handle chatbot queries with medical context and follow-up questions"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Use the medical chatbot if available
        if medical_chatbot:
            user_id = session.get('user_id', 'default_user')
            result = medical_chatbot.generate_medical_response(user_id, message)
            
            return jsonify({
                'response': result['response'],
                'follow_up_questions': result['follow_up_questions'],
                'message': message
            })
        else:
            # Fallback to the old RAG system
            search_results = rag_system.search_similar_chunks(message, top_k=3)
            
            if search_results['documents'] and search_results['documents'][0]:
                context_chunks = search_results['documents'][0]
                response = rag_system.generate_response(message, context_chunks, 'patient')
            else:
                response = "I'm here to help with your medical questions! Please ask me about your health, symptoms, medications, or any medical concerns you might have."
            
            return jsonify({
                'response': response,
                'follow_up_questions': [
                    "What health symptoms are you experiencing?",
                    "Do you have any medical conditions I should know about?",
                    "Are you currently taking any medications?"
                ],
                'message': message
            })
    
    except Exception as e:
        logger.error(f"Error processing chatbot message: {e}")
        return jsonify({'error': f'Error processing message: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check system status"""
    try:
        # Get collection count
        count = rag_system.collection.count()
        
        return jsonify({
            'status': 'healthy',
            'documents_stored': count,
            'embedding_model': 'all-MiniLM-L6-v2',
            'vector_db': 'ChromaDB',
            'llm_model': 'mistralai/mistral-7b-instruct',
            'symptom_scanner': 'available' if symptom_scanner else 'not_available',
            'medical_chatbot': 'available' if medical_chatbot else 'not_available'
        })
    
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return jsonify({'error': f'Error checking status: {str(e)}'}), 500

@app.route('/symptom-scanner/upload', methods=['POST'])
@login_required
def upload_symptom_image():
    """Upload and scan an image for symptoms using LLM analysis"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Check if it's an image file
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
            
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if not symptom_scanner:
                return jsonify({'error': 'Symptom scanner not available'}), 503
            
            # Scan the image with LLM
            scan_result = symptom_scanner.scan_image(filepath)
            
            # Get LLM analysis
            llm_analysis = scan_result.get('llm_analysis', {})
            structured_analysis = llm_analysis.get('structured_analysis', {})
            
            # Return only LLM analysis results
            result = {
                'llm_analysis': {
                    'raw_analysis': llm_analysis.get('llm_analysis', 'LLM analysis not available'),
                    'medical_relevance': structured_analysis.get('medical_relevance', 'Unknown'),
                    'primary_condition': structured_analysis.get('primary_condition', 'Unknown'),
                    'description': structured_analysis.get('description', 'No description available'),
                    'symptoms': structured_analysis.get('symptoms', 'Not specified'),
                    'severity': structured_analysis.get('severity', 'Unknown'),
                    'recommendations': structured_analysis.get('recommendations', 'Consult a healthcare professional'),
                    'confidence': structured_analysis.get('confidence', 'Unknown'),
                    'model_used': llm_analysis.get('model_used', 'placeholder')
                },
                'filename': filename
            }
            
            return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing symptom image: {e}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/symptom-scanner/scan-live', methods=['POST'])
@login_required
def scan_live_image():
    """Scan a live image from base64 data using LLM analysis"""
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image_data']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        if not symptom_scanner:
            return jsonify({'error': 'Symptom scanner not available'}), 503
        
        # Scan the image with LLM
        scan_result = symptom_scanner.scan_cv2_image(image)
        
        # Get LLM analysis
        llm_analysis = scan_result.get('llm_analysis', {})
        structured_analysis = llm_analysis.get('structured_analysis', {})
        
        # Return only LLM analysis results
        result = {
            'llm_analysis': {
                'raw_analysis': llm_analysis.get('llm_analysis', 'LLM analysis not available'),
                'medical_relevance': structured_analysis.get('medical_relevance', 'Unknown'),
                'primary_condition': structured_analysis.get('primary_condition', 'Unknown'),
                'description': structured_analysis.get('description', 'No description available'),
                'symptoms': structured_analysis.get('symptoms', 'Not specified'),
                'severity': structured_analysis.get('severity', 'Unknown'),
                'recommendations': structured_analysis.get('recommendations', 'Consult a healthcare professional'),
                'confidence': structured_analysis.get('confidence', 'Unknown'),
                'model_used': llm_analysis.get('model_used', 'placeholder')
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing live image: {e}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/symptom-scanner/condition-info/<condition>')
@login_required
def get_condition_info(condition):
    """Get detailed information about a specific condition"""
    try:
        if not symptom_scanner:
            return jsonify({'error': 'Symptom scanner not available'}), 503
        
        condition_info = symptom_scanner.get_condition_info(condition)
        return jsonify(condition_info)
        
    except Exception as e:
        logger.error(f"Error getting condition info: {e}")
        return jsonify({'error': f'Error getting condition info: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
