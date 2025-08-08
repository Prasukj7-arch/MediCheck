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

logger = logging.getLogger(__name__)

class SymptomScanner:
    def __init__(self, model_path: str = None):
        """
        Initialize the Symptom Scanner with a PyTorch model.
        
        Args:
            model_path: Path to the .pt model file. If None, will use a placeholder.
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = self._get_transforms()
        self.class_names = self._get_class_names()
        
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
    
    def scan_image(self, image_path: str) -> Dict[str, any]:
        """
        Scan an image for symptoms/diseases.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing predictions and metadata
        """
        try:
            # Preprocess the image
            image_tensor = self.preprocess_image(image_path)
            
            # Perform prediction
            predictions = self.predict(image_tensor)
            
            # Get top 3 predictions
            top_predictions = dict(list(predictions.items())[:3])
            
            # Determine the primary condition
            primary_condition = list(top_predictions.keys())[0]
            confidence = top_predictions[primary_condition]
            
            return {
                'primary_condition': primary_condition,
                'confidence': confidence,
                'all_predictions': predictions,
                'top_predictions': top_predictions,
                'image_path': image_path,
                'model_used': self.model_path or 'placeholder'
            }
        
        except Exception as e:
            logger.error(f"Error scanning image {image_path}: {e}")
            raise
    
    def scan_cv2_image(self, image: np.ndarray) -> Dict[str, any]:
        """
        Scan a CV2 image for symptoms/diseases.
        
        Args:
            image: CV2 image array
            
        Returns:
            Dictionary containing predictions and metadata
        """
        try:
            # Preprocess the image
            image_tensor = self.preprocess_cv2_image(image)
            
            # Perform prediction
            predictions = self.predict(image_tensor)
            
            # Get top 3 predictions
            top_predictions = dict(list(predictions.items())[:3])
            
            # Determine the primary condition
            primary_condition = list(top_predictions.keys())[0]
            confidence = top_predictions[primary_condition]
            
            return {
                'primary_condition': primary_condition,
                'confidence': confidence,
                'all_predictions': predictions,
                'top_predictions': top_predictions,
                'model_used': self.model_path or 'placeholder'
            }
        
        except Exception as e:
            logger.error(f"Error scanning CV2 image: {e}")
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
