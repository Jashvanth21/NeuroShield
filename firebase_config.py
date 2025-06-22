# firebase_config.py
import pyrebase

firebase_config = {
    "apiKey": "your-api-key",
    "authDomain": "your-app.firebaseapp.com",
    "projectId": "your-app-id",
    "storageBucket": "your-app.appspot.com",
    "messagingSenderId": "your-id",
    "appId": "your-app-id",
    "measurementId": "G-XXXX",
    "databaseURL": ""
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
