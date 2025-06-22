from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Data parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_WORKERS = 4
CLASSES = ['notumor', 'glioma', 'meningioma', 'pituitary']

# Model parameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20

# Normalization parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Firebase configuration
FIREBASE_CONFIG = {
    'storageBucket': 'your-bucket-name.appspot.com',
    'databaseURL': 'https://your-database-url.firebaseio.com'
}

# Model architecture
CNN_FEATURES = 2048  # ResNet50 features
SWIN_FEATURES = 1024  # Swin Transformer features
FUSION_FEATURES = 512
CLASSIFIER_FEATURES = 256

# Firebase collections
REPORTS_COLLECTION = 'reports'

# Firebase credentials path
FIREBASE_CRED_PATH = PROJECT_ROOT / 'firebase' / 'serviceAccountKey.json'

# Data augmentation parameters
AUGMENTATION_PROB = 0.5 