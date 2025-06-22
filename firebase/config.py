import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
from pathlib import Path

# Initialize Firebase Admin
def initialize_firebase():
    try:
        # Get the path to the service account key from environment variable
        cred_path = os.getenv('FIREBASE_CRED_PATH')
        if not cred_path:
            raise ValueError("FIREBASE_CRED_PATH environment variable not set")
        
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"Firebase service account key not found at {cred_path}")
        
        # Initialize the app
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        
        return True
    except Exception as e:
        print(f"Error initializing Firebase: {str(e)}")
        return False

# Get Firebase services
def get_auth():
    return auth

def get_firestore():
    return firestore.client() 