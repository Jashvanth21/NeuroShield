import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from models.hybrid_model import TumorClassifier
from utils.config import *

def load_model(model_path: str = str(MODEL_DIR / 'best_model.pth')) -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TumorClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict_image(model: nn.Module, image_path: str) -> tuple:
    device = next(model.parameters()).device
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs, _ = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return CLASSES[predicted_class], confidence

def main():
    # Load the model
    model = load_model()
    
    # Test directory containing images
    test_dir = DATA_DIR / 'Testing'
    
    # Process each class directory
    for class_name in CLASSES:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} does not exist")
            continue
            
        print(f"\nProcessing {class_name} images:")
        for img_path in class_dir.glob('*.jpg'):
            try:
                predicted_class, confidence = predict_image(model, str(img_path))
                print(f"Image: {img_path.name}")
                print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
                print(f"Actual: {class_name}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == '__main__':
    main() 