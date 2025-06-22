from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, url_for, flash, redirect
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from datetime import datetime
import uuid
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.hybrid_model import TumorClassifier
from utils.config import *
import json
import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

app = Flask(__name__, 
    static_url_path='',
    static_folder='src/static'
)

# Configure upload folder as an absolute path
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'uploads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TumorClassifier().to(device)
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get patient details from the form
        patient_details = {
            'name': request.form.get('patient_name', ''),
            'age': request.form.get('patient_age', ''),
            'gender': request.form.get('patient_gender', ''),
            'medical_record_number': request.form.get('medical_record_number', '')
        }
        
        if file and allowed_file(file.filename):
            try:
                # Open and validate image
                image = Image.open(file)
                is_valid, message, is_mri = validate_mri_image(image)
                
                if not is_valid:
                    return jsonify({'error': f'Invalid image: {message}'}), 400
                
                if not is_mri:
                    return jsonify({
                        'error': 'This appears to be a diagram or non-MRI image. Please upload an MRI scan for analysis.'
                    }), 400
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                file.seek(0)  # Reset file pointer after validation
                file.save(filepath)
                
                # Process image and get prediction
                result = process_image(filepath)
                
                # Generate report with patient details
                report_id = str(uuid.uuid4())
                report = {
                    'id': report_id,
                    'filename': filename,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'patient': patient_details,
                    'results': result
                }
                
                # Save report
                save_report(report)
                
                return jsonify(report)
                
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'}), 400
    
    return render_template('upload.html')

@app.route('/results/<report_id>')
@login_required
def results(report_id):
    report = load_report(report_id)
    if not report:
        return "Report not found", 404
    return render_template('results.html', report=report)

@app.route('/reports')
@login_required
def reports():
    reports_list = get_all_reports()
    return render_template('reports.html', reports=reports_list)

@app.route('/download_report/<report_id>')
@login_required
def download_report(report_id):
    try:
        report = load_report(report_id)
        if not report:
            return "Report not found", 404
        
        # Generate PDF
        pdf_path = generate_pdf(report)
        
        if not os.path.exists(pdf_path):
            return "Error generating PDF report", 500
        
        # Send the file with a proper filename
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"tumor_analysis_report_{report_id[:8]}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"Error in download_report: {e}")
        return f"Error generating report: {str(e)}", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(f"Attempting to serve file: {filename}")
    print(f"From directory: {app.config['UPLOAD_FOLDER']}")
    print(f"Full path: {os.path.join(app.config['UPLOAD_FOLDER'], filename)}")
    print(f"File exists: {os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename))}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/test_static')
def test_static():
    return """
    <h1>Static File Test</h1>
    <p>Upload folder: {}</p>
    <p>Upload folder exists: {}</p>
    <p>Upload folder contents: {}</p>
    """.format(
        app.config['UPLOAD_FOLDER'],
        os.path.exists(app.config['UPLOAD_FOLDER']),
        os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    )

@app.route('/delete_report/<report_id>', methods=['POST'])
@login_required
def delete_report(report_id):
    try:
        report = load_report(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
            
        # Delete the image file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], report['filename'])
        if os.path.exists(image_path):
            os.remove(image_path)
            
        # Delete the report file
        report_path = Path('reports') / f"{report_id}.json"
        if report_path.exists():
            os.remove(report_path)
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_all_reports', methods=['POST'])
@login_required
def delete_all_reports():
    try:
        # Delete all report files
        reports_dir = Path('reports')
        if reports_dir.exists():
            for report_file in reports_dir.glob('*.json'):
                # Load report to get image filename
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    
                # Delete associated image
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], report['filename'])
                if os.path.exists(image_path):
                    os.remove(image_path)
                    
                # Delete report file
                os.remove(report_file)
                
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'nii'}

def validate_mri_image(image):
    """
    Validate if the image is likely to be an MRI scan based on characteristics:
    1. Grayscale/similar RGB values (MRI scans are typically grayscale)
    2. Image intensity distribution
    3. Size and aspect ratio checks
    4. Edge and texture analysis
    Returns a tuple of (is_valid, message, is_mri)
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Initialize as not MRI by default
        is_mri = False
        
        # Check if image is grayscale or has similar RGB values (MRI characteristic)
        if len(img_array.shape) == 3:  # Color image
            # Calculate standard deviation between RGB channels
            rgb_std = np.std(img_array, axis=2).mean()
            if rgb_std > 15:  # Lower threshold for stricter check
                return True, "Valid image for analysis", False
            
            # Convert to grayscale for further analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Check image size and aspect ratio
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        if not (0.5 <= aspect_ratio <= 2.0):
            return True, "Valid image for analysis", False
            
        # Calculate histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        
        # Check intensity distribution
        # MRI typically has a specific intensity distribution
        peaks = np.where(np.diff(np.sign(np.diff(hist_norm))) < 0)[0] + 1
        if len(peaks) < 2:
            return True, "Valid image for analysis", False
            
        # Calculate edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # MRI scans typically have specific edge density ranges
        if edge_density < 0.01 or edge_density > 0.2:
            return True, "Valid image for analysis", False
            
        # Check for text-like content (common in diagrams/charts)
        # Use horizontal and vertical lines detection
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        vertical_lines = cv2.HoughLinesP(edges, 1, np.pi/2, 100, minLineLength=100, maxLineGap=10)
        
        if (horizontal_lines is not None and len(horizontal_lines) > 5) or \
           (vertical_lines is not None and len(vertical_lines) > 5):
            # Too many straight lines - likely a diagram
            return True, "Valid image for analysis", False
            
        # Calculate texture features using GLCM
        # MRI scans have specific texture patterns
        texture_window = gray[height//4:3*height//4, width//4:3*width//4]
        if texture_window.size > 0:
            std_dev = np.std(texture_window)
            if std_dev < 10:  # Too uniform for an MRI
                return True, "Valid image for analysis", False
        
        # If passed all checks, consider it an MRI
        is_mri = True
        return True, "Valid image for analysis", is_mri
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}", False

def get_tumor_size_category(diameter_mm):
    """
    Categorize tumor size based on diameter.
    """
    if diameter_mm < 20:
        return "Very Small"
    elif 20 <= diameter_mm < 30:
        return "Small"
    elif 30 <= diameter_mm < 50:
        return "Medium"
    elif 50 <= diameter_mm < 80:
        return "Large"
    else:
        return "Very Large"

def get_tumor_size_info(diameter_mm):
    """
    Get tumor size category and clinical significance based on diameter.
    """
    if diameter_mm < 20:
        return {
            "category": "Very Small",
            "clinical_significance": "Often asymptomatic, may be monitored"
        }
    elif 20 <= diameter_mm < 30:
        return {
            "category": "Small",
            "clinical_significance": "May cause symptoms depending on location"
        }
    elif 30 <= diameter_mm < 50:
        return {
            "category": "Medium",
            "clinical_significance": "Often requires treatment"
        }
    elif 50 <= diameter_mm < 80:
        return {
            "category": "Large",
            "clinical_significance": "Usually requires immediate intervention"
        }
    else:
        return {
            "category": "Very Large",
            "clinical_significance": "Critical condition requiring urgent treatment"
        }

def get_tumor_danger_level(tumor_type, size_category, confidence):
    """
    Determine the danger level and provide detailed risk assessment.
    """
    # Base danger points
    danger_points = 0
    
    # Size category contribution
    size_points = {
        "Very Small": 1,
        "Small": 2,
        "Medium": 3,
        "Large": 4,
        "Very Large": 5
    }
    
    # Tumor type contribution
    type_points = {
        "glioma": 5,  # Most aggressive
        "meningioma": 3,  # Moderate
        "pituitary": 2,  # Generally less aggressive
        "notumor": 0
    }
    
    # Calculate total points
    if size_category:
        danger_points += size_points.get(size_category, 0)
    danger_points += type_points.get(tumor_type.lower(), 0)
    
    # Adjust based on confidence
    confidence_factor = confidence if confidence > 0.5 else 0.5
    danger_points *= confidence_factor
    
    # Determine level and get details
    if danger_points >= 8:
        return {
            "level": "Critical",
            "color": "danger",
            "icon": "exclamation-triangle-fill",
            "details": "Immediate medical attention required. High-risk tumor detected with critical characteristics.",
            "recommendations": [
                "Urgent consultation with neurosurgical team",
                "Immediate treatment planning required",
                "Additional imaging may be needed for surgical planning",
                "Close monitoring of neurological symptoms"
            ]
        }
    elif danger_points >= 6:
        return {
            "level": "High Risk",
            "color": "warning",
            "icon": "exclamation-triangle",
            "details": "Serious condition requiring prompt medical attention.",
            "recommendations": [
                "Early consultation with specialist recommended",
                "Regular monitoring of tumor size",
                "Assessment of treatment options",
                "Regular neurological examinations"
            ]
        }
    elif danger_points >= 4:
        return {
            "level": "Moderate",
            "color": "info",
            "icon": "info-circle",
            "details": "Requires medical attention and monitoring.",
            "recommendations": [
                "Regular medical follow-up",
                "Periodic imaging to monitor growth",
                "Assessment of symptoms",
                "Discussion of treatment options"
            ]
        }
    else:
        return {
            "level": "Low",
            "color": "success",
            "icon": "check-circle",
            "details": "Lower risk but requires monitoring.",
            "recommendations": [
                "Regular check-ups",
                "Monitoring for any changes",
                "Baseline imaging for future comparison",
                "Patient education about symptoms to watch for"
            ]
        }

def process_image(image_path):
    try:
        # Open and validate image
        image = Image.open(image_path)
        is_valid, message, is_mri = validate_mri_image(image)
        
        if not is_valid:
            raise ValueError(message)
        
        # Continue with existing processing
        image = image.convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            classification, segmentation = model(image_tensor)
            probabilities = torch.nn.functional.softmax(classification, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Initialize tumor size dictionary
        tumor_size = {
            'area_mm2': None,
            'perimeter_mm': None,
            'diameter_mm': None,
            'size_category': None
        }
        
        # Only calculate tumor size if it's not "notumor" class
        predicted_class = CLASSES[predicted.item()]
        if predicted_class != "notumor":
            # Calculate tumor size with improved method
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image for size calculation")
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Normalize the image
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive thresholding instead of Otsu
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,  # Block size
                5    # C constant
            )
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by size
                min_area = img.shape[0] * img.shape[1] * 0.005  # 0.5% of image area
                max_area = img.shape[0] * img.shape[1] * 0.6    # Increased to 60% for large gliomas
                valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
                
                if valid_contours:
                    # Get the largest contour
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    
                    # Calculate pixel-to-mm ratio based on typical MRI parameters
                    # Standard brain width is approximately 180mm
                    brain_width_mm = 180  # Typical brain width in millimeters
                    pixel_to_mm = brain_width_mm / img.shape[1]
                    
                    # Calculate area in pixels and convert to mm²
                    area_pixels = cv2.contourArea(largest_contour)
                    area_mm2 = area_pixels * (pixel_to_mm ** 2)
                    
                    # Calculate perimeter in pixels and convert to mm
                    perimeter = cv2.arcLength(largest_contour, True)
                    correction_factor = 0.85
                    perimeter_mm = perimeter * pixel_to_mm * correction_factor
                    
                    # Calculate diameter
                    diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
                    
                    # Get size category and clinical significance
                    size_info = get_tumor_size_info(diameter_mm)
                    
                    # Update tumor size with rounded values
                    tumor_size.update({
                        'area_mm2': round(area_mm2, 2),
                        'perimeter_mm': round(perimeter_mm, 2),
                        'diameter_mm': round(diameter_mm, 2),
                        'size_category': size_info['category'],
                        'clinical_significance': size_info['clinical_significance']
                    })
        
        # Create the result dictionary
        result = {
            'tumor_type': predicted_class,
            'confidence': confidence.item(),
            'probabilities': {cls: prob.item() for cls, prob in zip(CLASSES, probabilities[0])},
            'tumor_size': tumor_size,
            'is_mri': is_mri
        }
        
        # Always include danger assessment
        if predicted_class == 'notumor':
            result['danger_assessment'] = {
                "level": "No Tumor Detected",
                "color": "success",
                "icon": "check-circle",
                "details": "No tumor was detected in the MRI scan. This indicates normal brain tissue appearance.",
                "recommendations": [
                    "Continue regular check-ups as recommended by your healthcare provider",
                    "Maintain a healthy lifestyle",
                    "Schedule follow-up imaging as recommended by your doctor",
                    "Report any new or concerning symptoms to your healthcare provider"
                ]
            }
        else:
            result['danger_assessment'] = get_tumor_danger_level(
                predicted_class,
                tumor_size.get('size_category'),
                confidence.item()
            )
        
        return result
        
    except Exception as e:
        print(f"Error in process_image: {e}")
        # Return basic result with default danger assessment
        return {
            'tumor_type': predicted_class,
            'confidence': confidence.item(),
            'probabilities': {cls: prob.item() for cls, prob in zip(CLASSES, probabilities[0])},
            'tumor_size': tumor_size,
            'is_mri': False,
            'danger_assessment': {
                "level": "Error in Analysis",
                "color": "warning",
                "icon": "exclamation-circle",
                "details": "There was an error processing some aspects of the image analysis.",
                "recommendations": [
                    "Please consult with your healthcare provider",
                    "Consider repeating the scan if recommended",
                    "Ensure the uploaded image is a valid MRI scan"
                ]
            }
        }

def save_report(report):
    # Create reports directory if it doesn't exist
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save report with proper encoding
    report_path = reports_dir / f"{report['id']}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

def load_report(report_id):
    try:
        report_path = Path('reports') / f"{report_id}.json"
        if not report_path.exists():
            return None
        
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}")
        return None

def get_all_reports():
    try:
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        
        for report_file in reports_dir.glob('*.json'):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    reports.append(report)
            except Exception as e:
                print(f"Error loading report {report_file}: {e}")
                continue
        
        # Sort reports by date, newest first
        return sorted(reports, key=lambda x: x['date'], reverse=True)
    except Exception as e:
        print(f"Error getting reports: {e}")
        return []

def generate_pdf(report):
    try:
        # Create reports directory in the src folder
        reports_dir = os.path.join(os.path.dirname(__file__), 'static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # PDF file path with absolute path using patient name
        patient_name = report.get('patient', {}).get('name', 'unknown')
        safe_name = "".join(x for x in patient_name if x.isalnum() or x.isspace()).strip()
        pdf_filename = f"{safe_name}_brain_tumor_report.pdf"
        pdf_path = os.path.join(reports_dir, pdf_filename)
        
        # Create the PDF document with adjusted margins
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=15,
            alignment=1,  # Center alignment
            textColor=colors.HexColor('#2c3e50')  # Dark blue color
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2c3e50')
        )
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=5,
            textColor=colors.HexColor('#34495e')
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=5,
            textColor=colors.HexColor('#2c3e50')
        )
        value_style = ParagraphStyle(
            'CustomValue',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=5,
            textColor=colors.HexColor('#2980b9')  # Blue color for values
        )
        label_style = ParagraphStyle(
            'CustomLabel',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=5,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold'
        )
        
        # Create the story (content)
        story = []
        
        # Title
        story.append(Paragraph("Brain Tumor Analysis Report", title_style))
        story.append(Spacer(1, 10))
        
        # Patient Information Section
        story.append(Paragraph("Patient Information", heading_style))
        patient = report.get('patient', {})
        story.append(Paragraph(f"<b>Name:</b> {patient.get('name', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Age:</b> {patient.get('age', 'N/A')} years", normal_style))
        story.append(Paragraph(f"<b>Gender:</b> {patient.get('gender', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Date:</b> {report['date']}", normal_style))
        story.append(Spacer(1, 15))
        
        # MRI Image Section
        story.append(Paragraph("MRI Scan", heading_style))
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], report['filename'])
        if os.path.exists(img_path):
            img = RLImage(img_path, width=300, height=300)
            story.append(img)
            story.append(Spacer(1, 15))
        
        # Primary Analysis Section
        story.append(Paragraph("Primary Analysis", heading_style))
        results = report['results']
        story.append(Paragraph(f"<b>Tumor Type:</b> {results['tumor_type'].title()}", normal_style))
        story.append(Paragraph(f"<b>Confidence:</b> {results['confidence']*100:.2f}%", normal_style))
        
        # Always show tumor size information if available, regardless of tumor type
        if results.get('tumor_size'):
            if results['tumor_size'].get('size_category'):
                story.append(Paragraph(f"<b>Size Category:</b> {results['tumor_size']['size_category']}", normal_style))
            if results['tumor_size'].get('clinical_significance'):
                story.append(Paragraph(f"<b>Clinical Significance:</b> {results['tumor_size']['clinical_significance']}", normal_style))
        
        story.append(Spacer(1, 15))
        
        # Tumor Measurements Section - Always show if available
        if results.get('tumor_size'):
            story.append(Paragraph("Tumor Measurements", heading_style))
            if results['tumor_size'].get('area_mm2'):
                story.append(Paragraph(f"<b>Area:</b> {results['tumor_size']['area_mm2']:.2f} mm²", normal_style))
            if results['tumor_size'].get('diameter_mm'):
                story.append(Paragraph(f"<b>Diameter:</b> {results['tumor_size']['diameter_mm']:.2f} mm", normal_style))
            if results['tumor_size'].get('perimeter_mm'):
                story.append(Paragraph(f"<b>Perimeter:</b> {results['tumor_size']['perimeter_mm']:.2f} mm", normal_style))
            story.append(Spacer(1, 15))
        
        # Risk Assessment Section - Show for all cases, including "notumor"
        story.append(Paragraph("Risk Assessment", heading_style))
        if results.get('danger_assessment'):
            danger = results['danger_assessment']
            story.append(Paragraph(f"<b>Risk Level:</b> {danger['level']}", normal_style))
            story.append(Paragraph(f"<b>Details:</b> {danger['details']}", normal_style))
            story.append(Spacer(1, 10))
            
            # Add recommendations with bullet points
            story.append(Paragraph("Recommendations:", subheading_style))
            for rec in danger['recommendations']:
                story.append(Paragraph(f"• {rec}", normal_style))
        else:
            # Default assessment for "notumor" case if no danger assessment is provided
            story.append(Paragraph("<b>Risk Level:</b> No Tumor Detected", normal_style))
            story.append(Paragraph("<b>Details:</b> No tumor was detected in the MRI scan. This is a normal finding.", normal_style))
            story.append(Spacer(1, 10))
            story.append(Paragraph("Recommendations:", subheading_style))
            story.append(Paragraph("• Continue regular check-ups as recommended by your healthcare provider", normal_style))
            story.append(Paragraph("• Maintain a healthy lifestyle", normal_style))
            story.append(Paragraph("• Schedule follow-up imaging as recommended by your doctor", normal_style))
        
        story.append(Spacer(1, 15))
        
        # Probability Distribution Section - Always show
        story.append(Paragraph("Probability Distribution", heading_style))
        for tumor_type, prob in results['probabilities'].items():
            story.append(Paragraph(f"<b>{tumor_type.title()}:</b> {prob*100:.2f}%", normal_style))
        
        # Build the PDF
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 