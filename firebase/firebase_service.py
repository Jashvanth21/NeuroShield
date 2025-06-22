from firebase_admin import auth, firestore
from datetime import datetime

class FirebaseService:
    def __init__(self):
        self.auth = auth
        self.db = firestore.client()

    def verify_token(self, token):
        """Verify Firebase ID token"""
        try:
            decoded_token = self.auth.verify_id_token(token)
            return decoded_token
        except Exception as e:
            print(f"Error verifying token: {str(e)}")
            return None

    def save_prediction(self, user_id, scan_id, prediction_data):
        """Save prediction results to Firestore"""
        try:
            # Create prediction document
            prediction_ref = self.db.collection('users').document(user_id).collection('predictions').document(scan_id)
            
            # Add timestamp
            prediction_data['timestamp'] = datetime.now()
            
            # Save to Firestore
            prediction_ref.set(prediction_data)
            
            return True
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return False

    def get_user_predictions(self, user_id):
        """Get all predictions for a user"""
        try:
            predictions_ref = self.db.collection('users').document(user_id).collection('predictions')
            predictions = predictions_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).get()
            
            return [pred.to_dict() for pred in predictions]
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            return []

    def delete_prediction(self, user_id, scan_id):
        """Delete a prediction"""
        try:
            prediction_ref = self.db.collection('users').document(user_id).collection('predictions').document(scan_id)
            prediction_ref.delete()
            return True
        except Exception as e:
            print(f"Error deleting prediction: {str(e)}")
            return False 