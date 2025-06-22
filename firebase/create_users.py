from firebase_admin import auth, firestore
from .config import initialize_firebase, get_auth, get_firestore
import time

def create_user(email, password, role):
    try:
        # Check if user already exists
        try:
            existing_user = auth.get_user_by_email(email)
            print(f"User {email} already exists. Updating role...")
            user_id = existing_user.uid
        except auth.UserNotFoundError:
            # Create new user
            user = auth.create_user(
                email=email,
                password=password,
                email_verified=True  # Important for production
            )
            user_id = user.uid
            print(f"Created new user: {email}")
        
        # Set custom claims (role)
        auth.set_custom_user_claims(user_id, {'role': role})
        
        # Create/update user document in Firestore
        db = firestore.client()
        user_ref = db.collection('users').document(user_id)
        
        # Use a transaction to ensure atomic update
        @firestore.transactional
        def update_user(transaction, user_ref):
            snapshot = user_ref.get(transaction=transaction)
            if snapshot.exists:
                transaction.update(user_ref, {
                    'role': role,
                    'updated_at': firestore.SERVER_TIMESTAMP
                })
            else:
                transaction.set(user_ref, {
                    'email': email,
                    'role': role,
                    'created_at': firestore.SERVER_TIMESTAMP
                })
        
        # Run the transaction
        transaction = db.transaction()
        update_user(transaction, user_ref)
        
        print(f"Successfully set up {role} user: {email}")
        return user_id
    except Exception as e:
        print(f"Error processing user {email}: {str(e)}")
        return None

if __name__ == "__main__":
    # Initialize Firebase
    if not initialize_firebase():
        print("Failed to initialize Firebase")
        exit(1)
    
    # Create sample users with more secure passwords
    users = [
        {
            'email': 'doctor@example.com',
            'password': 'Doctor@123',  # More secure password
            'role': 'doctor'
        },
        {
            'email': 'patient@example.com',
            'password': 'Patient@123',  # More secure password
            'role': 'patient'
        }
    ]
    
    # Add delay between user creations to avoid rate limiting
    for user_data in users:
        user_id = create_user(
            email=user_data['email'],
            password=user_data['password'],
            role=user_data['role']
        )
        if user_id:
            print(f"User ID: {user_id}")
        time.sleep(1)  # Add delay between creations 