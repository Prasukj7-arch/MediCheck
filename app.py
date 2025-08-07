import os
import json
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2
import io
import re
from sentence_transformers import SentenceTransformer
import chromadb
import openai
from dotenv import load_dotenv
import numpy as np
import requests

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

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
                    return "I apologize, but I can only answer questions related to medical reports and health information. Please ask me about your medical report, test results, medications, symptoms, or other health-related topics. For non-medical questions, I recommend consulting other appropriate resources."
                else:
                    return "I apologize, but I can only provide clinical analysis for medical-related queries. Please ask me about medical reports, clinical findings, diagnoses, treatments, or other healthcare-related topics. For non-medical questions, I recommend consulting other appropriate resources."
            
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            if perspective == 'patient':
                # Patient-friendly prompt
                prompt = f"""You are a compassionate medical expert helping a patient understand their medical report. 
                
                Medical Report Context:
                {context}
                
                Patient Question: {query}
                
                Please provide a clear, easy-to-understand response that:
                1. Uses simple, non-medical language when possible
                2. Explains medical terms in plain English
                3. Is reassuring but honest about any concerns
                4. Provides practical advice for the patient
                5. Encourages them to ask their doctor if they have concerns
                
                Remember: The patient may be anxious about their health, so be supportive and clear.
                
                Response:"""
                
                system_message = "You are a caring medical expert who helps patients understand their health information in simple, reassuring terms."
                
            else:
                # Doctor/clinical prompt
                prompt = f"""You are a senior medical expert analyzing a medical report for clinical decision-making. 
                
                Medical Report Context:
                {context}
                
                Clinical Question: {query}
                
                Please provide a detailed, professional clinical response that includes:
                1. Key clinical findings and their significance
                2. Relevant medical terminology and pathophysiology
                3. Differential diagnosis considerations
                4. Evidence-based treatment recommendations
                5. Follow-up and monitoring recommendations
                6. Any red flags or concerning findings
                
                Use appropriate medical terminology and clinical reasoning.
                
                Response:"""
                
                system_message = "You are a senior medical expert with extensive clinical experience, providing professional medical analysis and recommendations."
            
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

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
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
            'llm_model': 'mistralai/mistral-7b-instruct'
        })
    
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return jsonify({'error': f'Error checking status: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
