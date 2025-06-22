import cv2
import numpy as np
import albumentations as A
from typing import Tuple, Optional

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for model inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply augmentations
        augmented = self.transform(image=image)
        return augmented['image']

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better visualization.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image
        """
        # Apply CLAHE for better contrast
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced

    def segment_tumor(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform basic tumor segmentation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (segmentation mask, tumor area percentage)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Calculate tumor area percentage
        total_pixels = binary.shape[0] * binary.shape[1]
        tumor_pixels = np.sum(binary == 255)
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        return binary, tumor_percentage 