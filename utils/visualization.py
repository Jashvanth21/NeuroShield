import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import torch
from pathlib import Path

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = None
):
    """Plot training history including losses and accuracies."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()

def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    segmentation: np.ndarray,
    save_path: str = None
):
    """Visualize model prediction and segmentation."""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(prediction)
    plt.title('Prediction')
    plt.axis('off')
    
    # Segmentation
    plt.subplot(1, 3, 3)
    plt.imshow(segmentation, cmap='gray')
    plt.title('Segmentation')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()

def save_report_image(image: np.ndarray,
                     segmentation: np.ndarray,
                     classification: Dict[str, float],
                     patient_id: str,
                     report_id: str,
                     output_dir: str):
    """
    Save a comprehensive report image combining all visualizations.
    
    Args:
        image: Original input image
        segmentation: Segmentation mask
        classification: Dictionary of class probabilities
        patient_id: Patient identifier
        report_id: Report identifier
        output_dir: Directory to save the report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"report_{patient_id}_{report_id}.png"
    visualize_prediction(image, segmentation, classification, save_path)
    
    return str(save_path) 