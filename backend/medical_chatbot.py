import os
import json
import logging
import openai
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalChatbot:
    def __init__(self, openai_client=None):
        """
        Initialize the Medical Chatbot with conversation context management.
        
        Args:
            openai_client: OpenAI client for LLM integration
        """
        self.openai_client = openai_client
        self.conversation_history = {}  # Store conversation history per user
        self.max_history_length = 10  # Keep last 10 messages per user
        
    def is_medical_query(self, query: str) -> bool:
        """Check if the query is medical-related"""
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
            'prognosis', 'outcome', 'recovery', 'rehabilitation', 'therapy', 'counseling',
            'hurt', 'sick', 'unwell', 'feeling', 'body', 'stomach', 'head', 'chest',
            'back', 'arm', 'leg', 'hand', 'foot', 'neck', 'throat', 'mouth', 'nose',
            'ear', 'eye', 'skin', 'hair', 'nail', 'bone', 'muscle', 'joint'
        ]
        
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
    
    def add_to_history(self, user_id: str, message: str, sender: str):
        """Add message to conversation history"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'message': message,
            'sender': sender,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only the last max_history_length messages
        if len(self.conversation_history[user_id]) > self.max_history_length:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history_length:]
    
    def get_conversation_context(self, user_id: str) -> str:
        """Get conversation context for the user"""
        if user_id not in self.conversation_history:
            return ""
        
        context = []
        for msg in self.conversation_history[user_id]:
            role = "User" if msg['sender'] == 'user' else "Assistant"
            context.append(f"{role}: {msg['message']}")
        
        return "\n".join(context)
    
    def generate_medical_response(self, user_id: str, message: str) -> Dict[str, str]:
        """
        Generate a medical response with context and follow-up questions.
        
        Args:
            user_id: Unique identifier for the user
            message: User's message
            
        Returns:
            Dictionary containing response and follow-up questions
        """
        try:
            if not self.openai_client:
                return self._get_placeholder_response()
            
            # Check if it's a medical query
            if not self.is_medical_query(message):
                return {
                    'response': "I'm a medical assistant and can only help with health-related questions. Please ask me about symptoms, medications, medical conditions, or any health concerns you might have.",
                    'follow_up_questions': [
                        "What health symptoms are you experiencing?",
                        "Do you have any medical conditions I should know about?",
                        "Are you currently taking any medications?"
                    ]
                }
            
            # Get conversation context
            context = self.get_conversation_context(user_id)
            
            # Create the prompt for medical conversation
            prompt = f"""You are a knowledgeable and empathetic medical assistant. You help patients understand their health concerns and provide guidance while always encouraging them to consult healthcare professionals for proper diagnosis and treatment.

IMPORTANT: You are NOT a doctor and cannot provide medical diagnosis. Always recommend consulting healthcare professionals for serious concerns.

Conversation Context:
{context}

Current User Message: {message}

Please provide a helpful, informative response that:
1. Addresses the user's medical question or concern
2. Uses simple, clear language
3. Provides general information and guidance
4. Encourages consulting healthcare professionals when appropriate
5. Shows empathy and understanding
6. Generates 2-3 relevant follow-up questions to better understand their situation

Your response should be conversational and helpful, but always remind users that you're an AI assistant and they should consult healthcare professionals for medical advice.

Response:"""

            # Call the LLM
            response = self.openai_client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant who provides general health information and guidance while always encouraging users to consult healthcare professionals for proper medical advice."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Extract follow-up questions from the response
            follow_up_questions = self._extract_follow_up_questions(llm_response)
            
            # Clean up the response (remove follow-up questions from main response)
            clean_response = self._clean_response(llm_response)
            
            # Add to conversation history
            self.add_to_history(user_id, message, 'user')
            self.add_to_history(user_id, clean_response, 'assistant')
            
            return {
                'response': clean_response,
                'follow_up_questions': follow_up_questions
            }
            
        except Exception as e:
            logger.error(f"Error generating medical response: {e}")
            return self._get_placeholder_response()
    
    def _extract_follow_up_questions(self, response: str) -> List[str]:
        """Extract follow-up questions from the response"""
        questions = []
        
        # Look for questions in the response
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.endswith('?') and len(line) > 10:
                # Remove common prefixes
                clean_line = line
                for prefix in ['Follow-up:', 'Question:', 'Also:', 'Additionally:']:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):].strip()
                        break
                questions.append(clean_line)
        
        # If no questions found, generate some generic ones
        if not questions:
            questions = [
                "What symptoms are you experiencing?",
                "How long have you been feeling this way?",
                "Have you consulted a healthcare professional about this?"
            ]
        
        return questions[:3]  # Return max 3 questions
    
    def _clean_response(self, response: str) -> str:
        """Clean the response by removing follow-up questions section"""
        # Remove any section that starts with "Follow-up questions:" or similar
        lines = response.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['follow-up', 'questions:', 'additional questions']):
                break
            if line:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()
    
    def _get_placeholder_response(self) -> Dict[str, str]:
        """Get placeholder response when LLM is not available"""
        return {
            'response': "I'm here to help with your medical questions! Please ask me about your health, symptoms, medications, or any medical concerns you might have. Remember to always consult with healthcare professionals for proper medical advice.",
            'follow_up_questions': [
                "What health symptoms are you experiencing?",
                "Do you have any medical conditions I should know about?",
                "Are you currently taking any medications?"
            ]
        }
