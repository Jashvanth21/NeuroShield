import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.hybrid_model import TumorClassifier
from utils.config import *
from train import BrainTumorDataset

def evaluate_model():
    """
    Evaluate the trained model on the test dataset.
    """
    try:
        print("Starting model evaluation...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Check if model file exists
        model_path = MODEL_DIR / 'best_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Create model and load trained weights
        print("\nLoading model...")
        model = TumorClassifier().to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Create test dataset and dataloader
        print("\nCreating test dataset with 150 images per class...")
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        
        test_dir = DATA_DIR / 'Testing'
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found at {test_dir}")
            
        test_dataset = BrainTumorDataset(test_dir, transform=transform, max_images_per_class=150)
        if len(test_dataset) == 0:
            raise ValueError("No images found in test dataset")
            
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS)
        
        # Initialize lists to store predictions and true labels
        all_preds = []
        all_labels = []
        
        # Evaluate model
        print("\nStarting evaluation...")
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    classification = outputs[0]
                else:
                    classification = outputs
                _, preds = torch.max(classification, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=CLASSES))
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES,
                    yticklabels=CLASSES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(MODEL_DIR / 'confusion_matrix.png')
        plt.close()
        
        # Calculate overall accuracy
        accuracy = np.mean(all_preds == all_labels)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Save results to file
        with open(MODEL_DIR / 'evaluation_results.txt', 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("=======================\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(all_labels, all_preds, target_names=CLASSES))
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
        
        print("\nEvaluation complete! Results saved to:")
        print(f"- Confusion Matrix: {MODEL_DIR / 'confusion_matrix.png'}")
        print(f"- Detailed Results: {MODEL_DIR / 'evaluation_results.txt'}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    evaluate_model() 