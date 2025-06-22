import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import argparse
from typing import Dict, Tuple
from pathlib import Path
import logging

from models.hybrid_model import TumorDetectionSystem
from preprocessing.image_processor import ImagePreprocessor
from firebase.firebase_manager import FirebaseManager
from utils.config import *
from utils.visualization import save_report_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorAnalyzer:
    def __init__(self, model_path: str, firebase_cred_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TumorDetectionSystem(model_path)
        self.preprocessor = ImagePreprocessor(target_size=IMAGE_SIZE)
        self.firebase = FirebaseManager(firebase_cred_path)
        
        self.classes = CLASSES
        
    def analyze_image(self, image_path: str, patient_id: str) -> Dict:
        """
        Analyze a brain MRI image and generate a report.
        
        Args:
            image_path: Path to the input image
            patient_id: Unique identifier for the patient
            
        Returns:
            Dictionary containing analysis results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Enhance image
        enhanced_image = self.preprocessor.enhance(image_np)
        
        # Preprocess for model
        processed_image = self.preprocessor.preprocess(enhanced_image)
        processed_image = torch.from_numpy(processed_image).unsqueeze(0)
        
        # Get model predictions
        classification, segmentation = self.model.predict(processed_image)
        
        # Process classification results
        probs = torch.softmax(classification, dim=1)[0]
        class_probs = {cls: float(prob) for cls, prob in zip(self.classes, probs)}
        predicted_class = self.classes[torch.argmax(probs).item()]
        
        # Process segmentation
        segmentation = segmentation.squeeze().cpu().numpy()
        _, tumor_size = self.preprocessor.segment_tumor(enhanced_image)
        
        # Convert segmentation mask to base64
        mask_image = Image.fromarray((segmentation * 255).astype(np.uint8))
        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Generate report
        report = {
            'patient_id': patient_id,
            'image_path': image_path,
            'classification': class_probs,
            'predicted_class': predicted_class,
            'tumor_size': float(tumor_size),
            'segmentation_mask': mask_base64
        }
        
        # Store report in Firebase
        report_id = self.firebase.store_report(
            patient_id=patient_id,
            image_path=image_path,
            classification=class_probs,
            tumor_size=float(tumor_size),
            segmentation_mask=mask_base64
        )
        
        report['report_id'] = report_id
        
        # Save visualization
        report_image_path = save_report_image(
            image=image_np,
            segmentation=segmentation,
            classification=class_probs,
            patient_id=patient_id,
            report_id=report_id,
            output_dir=str(PROJECT_ROOT / 'reports')
        )
        
        report['report_image_path'] = report_image_path
        return report

def process_image(image_path):
    """
    Process an MRI image and return prediction results.
    
    Args:
        image_path (str): Path to the MRI image
        
    Returns:
        dict: Prediction results including class, confidence, and tumor size
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # Resize to expected input size
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # Change to C, H, W
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Load model (you need to have your trained model file)
        model_path = Path('src/models/best_model.pth')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Map prediction to class names
        classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
        predicted_class = classes[predicted.item()]
        confidence = confidence.item()
        
        # Calculate tumor size (this is a simplified estimation)
        # In a real application, you would use segmentation for this
        tumor_size = 0.0
        if predicted_class != 'No Tumor':
            # This is a placeholder - replace with actual tumor size calculation
            tumor_size = confidence * 100  # Using confidence as a proxy for size
        
        # Get probabilities for all classes
        class_probabilities = {
            cls: float(prob) 
            for cls, prob in zip(classes, probabilities[0])
        }
        
        return {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'tumor_size': float(tumor_size),
            'probabilities': class_probabilities
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Brain Tumor Analysis')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--patient_id', type=str, required=True,
                      help='Patient ID')
    parser.add_argument('--model_path', type=str, 
                      default=str(MODEL_DIR / 'best_model.pth'),
                      help='Path to the trained model')
    parser.add_argument('--firebase_cred', type=str,
                      default=str(FIREBASE_CRED_PATH),
                      help='Path to Firebase credentials')
    
    args = parser.parse_args()
    
    # Create reports directory
    (PROJECT_ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    
    analyzer = BrainTumorAnalyzer(args.model_path, args.firebase_cred)
    results = analyzer.analyze_image(args.image_path, args.patient_id)
    
    print("\nAnalysis Results:")
    print(f"Patient ID: {results['patient_id']}")
    print(f"Predicted Class: {results['predicted_class']}")
    print("\nClassification Probabilities:")
    for cls, prob in results['classification'].items():
        print(f"{cls}: {prob:.2%}")
    print(f"\nTumor Size: {results['tumor_size']:.2f}%")
    print(f"Report ID: {results['report_id']}")
    print(f"Report Image: {results['report_image_path']}")

if __name__ == '__main__':
    main() 