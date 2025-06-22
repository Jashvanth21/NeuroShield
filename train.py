import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
from pathlib import Path
import traceback
import gc

from models.hybrid_model import TumorClassifier
from preprocessing.image_processor import ImagePreprocessor
from utils.config import *
from utils.visualization import plot_training_history

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, max_images_per_class: int = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        # Load images and labels for each class
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue

            # Get all images for this class
            class_images = []
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    class_images.append(os.path.join(class_dir, img_name))

            # If max_images_per_class is specified, randomly sample that many images
            if max_images_per_class is not None and len(class_images) > max_images_per_class:
                indices = torch.randperm(len(class_images))[:max_images_per_class]
                class_images = [class_images[i] for i in indices]

            # Add the selected images and their labels
            self.images.extend(class_images)
            self.labels.extend([self.class_to_idx[class_name]] * len(class_images))

            print(f"Loaded {len(class_images)} images from {class_name} class")

        print(f"Total loaded: {len(self.images)} images from {root_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        return image, label

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: str = 'cuda'
) -> Tuple[nn.Module, List[float], List[float], List[float], List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stopping_patience = 5
    early_stopping_counter = 0

    # Initialize history lists
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Initialize model with a dummy batch
    print("Initializing model with a dummy batch...")
    dummy_batch = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        model(dummy_batch)
    print("Model initialization complete!")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        try:
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                try:
                    outputs, _ = model(images)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    print(f"Image shape: {images.shape}")
                    # Print model architecture for debugging
                    print(model)
                    raise e

            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total

            # Store history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_DIR / 'best_model.pth')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        except Exception as e:
            print(f"Error during epoch {epoch+1}: {e}")
            traceback.print_exc()
            # Save the model even if there's an error
            torch.save(model.state_dict(), MODEL_DIR / f'error_model_epoch_{epoch+1}.pth')
            raise e

    return model, train_losses, val_losses, train_accs, val_accs

def main():
    try:
        # Create necessary directories
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Data transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=AUGMENTATION_PROB),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

        # Validation transform (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

        print("\nCreating datasets with images per class...")
        train_dataset = BrainTumorDataset(DATA_DIR / 'Training', transform=transform, max_images_per_class=500)
        val_dataset = BrainTumorDataset(DATA_DIR / 'Testing', transform=val_transform, max_images_per_class=350)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Error: No images found in the dataset")
            return

        print(f"Training dataset size: {len(train_dataset)} images")
        print(f"Validation dataset size: {len(val_dataset)} images")

        # Create data loaders with CPU-optimized settings
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                num_workers=2)  # Smaller batch size and fewer workers for CPU
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                               num_workers=2)  # Smaller batch size and fewer workers for CPU

        # Initialize model
        model = TumorClassifier().to(device)
        print("Model created successfully")

        # Train model
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, num_epochs=30, device=device
        )

        # Plot and save training history
        plot_training_history(
            train_losses, val_losses, train_accs, val_accs,
            save_path=str(MODEL_DIR / 'training_history.png')
        )

        print("Training completed!")

    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        # Clean up memory
        gc.collect()

if __name__ == '__main__':
    main()
