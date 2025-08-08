import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import json
import base64
import openai

logger = logging.getLogger(__name__)

class SymptomScanner:
    def __init__(self, model_path: str = None, openai_client=None):
        """
        Initialize the Symptom Scanner with a PyTorch model and LLM integration.
        
        Args:
            model_path: Path to the .pt model file. If None, will use a placeholder.
            openai_client: OpenAI client for LLM integration
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = self._get_transforms()
        self.class_names = self._get_class_names()
        self.openai_client = openai_client
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning(f"Model path {model_path} not found. Using placeholder model.")
            self._create_placeholder_model()
    
    def _get_transforms(self) -> transforms.Compose:
        """Get the image transformations for the model."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_class_names(self) -> List[str]:
        """Get the class names for the model predictions."""
        # This is a placeholder. In a real implementation, you would load this from a file
        # or define it based on your specific model's classes
        return [
            "Normal", "Acne", "Eczema", "Psoriasis", "Rosacea", "Dermatitis",
            "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma",
            "Fungal Infection", "Bacterial Infection", "Viral Infection",
            "Allergic Reaction", "Insect Bite", "Burn", "Wound", "Rash",
            "Hives", "Vitiligo", "Alopecia"
        ]
    
    def _create_placeholder_model(self):
        """Create a placeholder model for testing purposes."""
        class PlaceholderModel(nn.Module):
            def __init__(self, num_classes: int = 20):
                super(PlaceholderModel, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        self.model = PlaceholderModel(len(self.class_names))
        self.model.to(self.device)
        self.model.eval()
        logger.info("Placeholder model created and loaded.")
    
    def load_model(self, model_path: str):
        """Load a PyTorch model from the given path."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the model
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_placeholder_model()
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            return image_tensor.to(self.device)
        
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def preprocess_cv2_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a CV2 image for model inference.
        
        Args:
            image: CV2 image array
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert CV2 image to PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Apply transformations
            image_tensor = self.transform(image_pil)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            return image_tensor.to(self.device)
        
        except Exception as e:
            logger.error(f"Error preprocessing CV2 image: {e}")
            raise
    
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Perform inference on the preprocessed image.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary with class names and their probabilities
        """
        try:
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Convert to dictionary
                predictions = {}
                for i, prob in enumerate(probabilities[0]):
                    predictions[self.class_names[i]] = prob.item()
                
                # Sort by probability (descending)
                sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
                
                return sorted_predictions
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def analyze_image_with_llm(self, image_path: str = None, image_array: np.ndarray = None, image_base64: str = None) -> Dict[str, any]:
        """
        Analyze an image using the LLM to provide medical descriptions.
        
        Args:
            image_path: Path to the image file (optional)
            image_array: CV2 image array (optional)
            image_base64: Base64 encoded image string (optional)
            
        Returns:
            Dictionary containing LLM analysis results
        """
        try:
            if not self.openai_client:
                logger.warning("OpenAI client not available. Using placeholder analysis.")
                return self._get_placeholder_llm_analysis()
            
            # Prepare image data for LLM
            if image_path:
                # Read image file and convert to base64
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
            elif image_array is not None:
                # Convert CV2 array to base64
                _, buffer = cv2.imencode('.jpg', image_array)
                image_data = base64.b64encode(buffer).decode('utf-8')
            elif image_base64:
                # Use provided base64 data
                if image_base64.startswith('data:image'):
                    image_data = image_base64.split(',')[1]
                else:
                    image_data = image_base64
            else:
                raise ValueError("No image data provided")
            
            # Create the prompt for medical analysis
            prompt = """You are a medical expert analyzing an image. Please examine the image carefully and provide a detailed medical analysis.

Please provide your analysis in the following format:

1. **Medical Relevance**: Is this image medically relevant? (Yes/No)
2. **Primary Condition**: What is the primary medical condition or finding?
3. **Description**: Provide a detailed medical description of what you observe
4. **Symptoms**: List any visible symptoms or signs
5. **Severity**: Assess the severity (None, Mild, Moderate, Severe, Critical)
6. **Recommendations**: Provide medical recommendations or next steps
7. **Confidence**: Rate your confidence in this analysis (0-100%)

If the image is not medically relevant, please state "Not associated with medical conditions" and provide a brief description of what you see.

Please be thorough and professional in your analysis."""

            # Try different vision models available on OpenRouter
            vision_models = [
                "openai/gpt-4o",
                "openai/gpt-4o-mini", 
                "anthropic/claude-3-5-sonnet",
                "anthropic/claude-3-haiku",
                "google/gemini-pro-1.5"
            ]
            
            response = None
            model_used = None
            
            for model in vision_models:
                try:
                    logger.info(f"Trying vision model: {model}")
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    model_used = model
                    break
                except Exception as e:
                    logger.warning(f"Model {model} failed: {e}")
                    continue
            
            if response is None:
                logger.error("All vision models failed. Using text-only analysis.")
                # Fallback to text-only analysis
                return self._analyze_image_text_only(image_data)
            
            # Parse the response
            llm_response = response.choices[0].message.content.strip()
            
            # Extract structured information from the response
            analysis = self._parse_llm_response(llm_response)
            
            return {
                'llm_analysis': llm_response,
                'structured_analysis': analysis,
                'model_used': model_used
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image with LLM: {e}")
            return self._get_placeholder_llm_analysis()
    
    def _analyze_image_text_only(self, image_data: str) -> Dict[str, any]:
        """
        Fallback method for text-only analysis when vision models are not available.
        """
        try:
            # Create a text-only prompt
            prompt = f"""You are a medical expert. I have an image that I cannot directly show you, but I can describe it as a medical image that needs analysis.

Please provide a general medical analysis template that would be appropriate for medical image analysis:

1. **Medical Relevance**: Based on typical medical images, this is likely medically relevant
2. **Primary Condition**: This would require visual examination to determine
3. **Description**: Medical images typically show anatomical structures, pathological findings, or diagnostic information
4. **Symptoms**: Would depend on the specific condition visible in the image
5. **Severity**: Would be assessed based on the findings
6. **Recommendations**: General recommendation would be to consult with a healthcare professional for proper diagnosis
7. **Confidence**: Cannot provide confidence without visual analysis

Please note: This is a general analysis as the actual image cannot be processed. For accurate diagnosis, please consult with a healthcare professional."""

            response = self.openai_client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",  # Use text-only model
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            return {
                'llm_analysis': llm_response,
                'structured_analysis': {
                    'medical_relevance': 'Likely relevant (requires visual confirmation)',
                    'primary_condition': 'Requires visual examination',
                    'description': 'Medical image analysis requires visual examination',
                    'symptoms': 'Cannot determine without visual analysis',
                    'severity': 'Cannot assess without visual analysis',
                    'recommendations': 'Consult with a healthcare professional for proper diagnosis',
                    'confidence': 'Cannot provide confidence without visual analysis'
                },
                'model_used': 'mistralai/mistral-7b-instruct (text-only)'
            }
            
        except Exception as e:
            logger.error(f"Error in text-only analysis: {e}")
            return self._get_placeholder_llm_analysis()
    
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """
        Parse the LLM response to extract structured information.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Structured analysis dictionary
        """
        try:
            analysis = {
                'medical_relevance': 'Unknown',
                'primary_condition': 'Unknown',
                'description': response,
                'symptoms': 'Not specified',
                'severity': 'Unknown',
                'recommendations': 'Consult a healthcare professional',
                'confidence': 'Unknown'
            }
            
            # Try to extract structured information from the response
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if 'medical relevance' in line.lower():
                    if 'yes' in line.lower():
                        analysis['medical_relevance'] = 'Yes'
                    elif 'no' in line.lower():
                        analysis['medical_relevance'] = 'No'
                elif 'primary condition' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        analysis['primary_condition'] = parts[1].strip()
                elif 'symptoms' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        analysis['symptoms'] = parts[1].strip()
                elif 'severity' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        analysis['severity'] = parts[1].strip()
                elif 'recommendations' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        analysis['recommendations'] = parts[1].strip()
                elif 'confidence' in line.lower():
                    # Extract percentage
                    import re
                    confidence_match = re.search(r'(\d+)%', line)
                    if confidence_match:
                        analysis['confidence'] = f"{confidence_match.group(1)}%"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                'medical_relevance': 'Unknown',
                'primary_condition': 'Unknown',
                'description': response,
                'symptoms': 'Not specified',
                'severity': 'Unknown',
                'recommendations': 'Consult a healthcare professional',
                'confidence': 'Unknown'
            }
    
    def _get_placeholder_llm_analysis(self) -> Dict[str, any]:
        """Get placeholder analysis when LLM is not available."""
        return {
            'llm_analysis': 'LLM analysis is not available. Please ensure your OPENROUTER_API_KEY is properly configured and you have access to vision-capable models like GPT-4o, Claude-3, or Gemini Pro.',
            'structured_analysis': {
                'medical_relevance': 'Cannot determine without LLM analysis',
                'primary_condition': 'Requires LLM analysis',
                'description': 'LLM analysis is required for detailed medical image analysis. Please check your API configuration.',
                'symptoms': 'Cannot determine without LLM analysis',
                'severity': 'Cannot assess without LLM analysis',
                'recommendations': 'Please ensure LLM integration is properly configured for accurate analysis',
                'confidence': 'Cannot provide confidence without LLM analysis'
            },
            'model_used': 'placeholder (LLM not available)'
        }
    
    def scan_image(self, image_path: str) -> Dict[str, any]:
        """
        Scan an image for symptoms/diseases using LLM analysis only.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing LLM analysis results
        """
        try:
            # Get LLM analysis
            llm_analysis = self.analyze_image_with_llm(image_path=image_path)
            
            return {
                'llm_analysis': llm_analysis
            }
        
        except Exception as e:
            logger.error(f"Error scanning image {image_path}: {e}")
            raise
    
    def scan_cv2_image(self, image: np.ndarray) -> Dict[str, any]:
        """
        Scan a CV2 image for symptoms/diseases using LLM analysis only.
        
        Args:
            image: CV2 image array
            
        Returns:
            Dictionary containing LLM analysis results
        """
        try:
            # Get LLM analysis
            llm_analysis = self.analyze_image_with_llm(image_array=image)
            
            return {
                'llm_analysis': llm_analysis
            }
        
        except Exception as e:
            logger.error(f"Error scanning CV2 image: {e}")
            raise
    
    def scan_image_base64(self, image_base64: str) -> Dict[str, any]:
        """
        Scan a base64 encoded image for symptoms/diseases using LLM analysis only.
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            Dictionary containing LLM analysis results
        """
        try:
            # Decode base64 to numpy array
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            image_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image data")
            
            # Get LLM analysis
            llm_analysis = self.analyze_image_with_llm(image_base64=image_base64)
            
            return {
                'llm_analysis': llm_analysis
            }
        
        except Exception as e:
            logger.error(f"Error scanning base64 image: {e}")
            raise
    
    def get_condition_info(self, condition: str) -> Dict[str, str]:
        """
        Get information about a specific condition.
        
        Args:
            condition: Name of the condition
            
        Returns:
            Dictionary containing information about the condition
        """
        # This is a placeholder. In a real implementation, you would have a database
        # or API call to get detailed information about conditions
        condition_info = {
            "Normal": {
                "description": "No significant skin conditions detected.",
                "symptoms": "Healthy skin appearance.",
                "recommendations": "Continue with regular skin care routine.",
                "severity": "None"
            },
            "Acne": {
                "description": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
                "symptoms": "Whiteheads, blackheads, pimples, large painful bumps under the skin.",
                "recommendations": "Keep skin clean, avoid touching face, use non-comedogenic products, consider seeing a dermatologist for severe cases.",
                "severity": "Mild to Moderate"
            },
            "Eczema": {
                "description": "A condition that makes your skin red and itchy.",
                "symptoms": "Dry, sensitive skin, intense itching, red patches, rough or scaly patches.",
                "recommendations": "Moisturize regularly, avoid harsh soaps, use gentle skin care products, consider prescription treatments.",
                "severity": "Mild to Severe"
            },
            "Psoriasis": {
                "description": "A skin disorder that causes skin cells to multiply up to 10 times faster than normal.",
                "symptoms": "Red patches of skin covered with thick, silvery scales, dry, cracked skin, itching or burning.",
                "recommendations": "See a dermatologist for proper diagnosis and treatment, avoid triggers, keep skin moisturized.",
                "severity": "Moderate to Severe"
            },
            "Rosacea": {
                "description": "A chronic skin condition that causes redness and visible blood vessels in your face.",
                "symptoms": "Facial redness, swollen red bumps, eye problems, enlarged nose.",
                "recommendations": "Avoid triggers (sun, spicy foods, alcohol), use gentle skin care, see a dermatologist for treatment.",
                "severity": "Mild to Moderate"
            }
        }
        
        return condition_info.get(condition, {
            "description": f"Information about {condition} is not available.",
            "symptoms": "Consult a healthcare professional for accurate diagnosis.",
            "recommendations": "Please see a doctor for proper medical advice.",
            "severity": "Unknown"
        })
