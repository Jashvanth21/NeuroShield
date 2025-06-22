import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from typing import Dict, Any, List, Optional

class FirebaseManager:
    def __init__(self, cred_path: str):
        # Initialize Firebase
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def store_report(self, 
                    patient_id: str,
                    image_path: str,
                    classification: Dict[str, float],
                    tumor_size: float,
                    segmentation_mask: str) -> str:
        """
        Store analysis report in Firebase.
        
        Args:
            patient_id: Unique identifier for the patient
            image_path: Path to the analyzed image
            classification: Dictionary of class probabilities
            tumor_size: Estimated tumor size in percentage
            segmentation_mask: Base64 encoded segmentation mask
            
        Returns:
            Document ID of the stored report
        """
        report_data = {
            'patient_id': patient_id,
            'image_path': image_path,
            'classification': classification,
            'tumor_size': tumor_size,
            'segmentation_mask': segmentation_mask,
            'timestamp': datetime.now(),
            'status': 'completed'
        }
        
        # Add to reports collection
        doc_ref = self.db.collection('reports').document()
        doc_ref.set(report_data)
        
        return doc_ref.id

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific report by ID.
        
        Args:
            report_id: Document ID of the report
            
        Returns:
            Report data if found, None otherwise
        """
        doc_ref = self.db.collection('reports').document(report_id)
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        return None

    def get_patient_reports(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all reports for a specific patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List of patient's reports
        """
        reports = self.db.collection('reports')\
            .where('patient_id', '==', patient_id)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .stream()
            
        return [doc.to_dict() for doc in reports]

    def update_report_status(self, report_id: str, status: str) -> bool:
        """
        Update the status of a report.
        
        Args:
            report_id: Document ID of the report
            status: New status to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_ref = self.db.collection('reports').document(report_id)
            doc_ref.update({'status': status})
            return True
        except Exception:
            return False 