# Brain Tumor Detection System

A comprehensive web-based application for detecting and analyzing brain tumors from MRI scans using deep learning.

## Features

- **Advanced Tumor Detection**: Utilizes a hybrid deep learning model to detect and classify brain tumors
- **Multi-Class Classification**: Identifies different types of brain tumors (Glioma, Meningioma, Pituitary, No Tumor)
- **Detailed Analysis**: Provides comprehensive measurements including:
  - Tumor size (area, diameter, perimeter)
  - Confidence scores
  - Risk assessment
  - Clinical recommendations
- **User Management**: Secure login and registration system
- **Report Management**: Save and manage patient reports
- **PDF Report Generation**: Generate detailed PDF reports with analysis results
- **Responsive UI**: Modern, user-friendly interface
- **Image Validation**: Validates uploaded images to ensure they are proper MRI scans

## Technology Stack

- **Backend**: Python, Flask
- **Database**: SQLite
- **Deep Learning**: PyTorch
- **Image Processing**: OpenCV, PIL
- **Frontend**: HTML5, CSS3, Bootstrap
- **PDF Generation**: ReportLab
- **Authentication**: Flask-Login

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/braintumor.git
cd braintumor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

## Configuration

1. Create a `.env` file in the root directory:
```env
FLASK_APP=src/app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
```

2. Configure the upload directory in `src/app.py`:
```python
app.config['UPLOAD_FOLDER'] = 'path/to/upload/folder'
```

## Usage

1. Start the application:
```bash
flask run
```

2. Access the application at `http://localhost:5000`

3. Register a new account or login with existing credentials

4. Upload an MRI scan for analysis

## Project Structure

```
braintumor/
├── src/
│   ├── app.py              # Main application file
│   ├── models/             # Neural network models
│   ├── static/             # Static files (CSS, JS, images)
│   ├── templates/          # HTML templates
│   ├── utils/             # Utility functions
│   └── preprocessing/      # Image preprocessing modules
├── models/                # Trained model weights
├── data/                  # Training data
├── reports/              # Generated reports
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Model Architecture

The system uses a hybrid deep learning model that combines:
- Feature extraction using a pre-trained CNN
- Custom classification layers for tumor detection
- Image segmentation for precise measurements

## Security Features

- Password hashing
- Session management
- Secure file uploads
- Input validation
- Error handling

## Report Generation

The system generates comprehensive PDF reports including:
- Patient information
- MRI scan visualization
- Tumor measurements
- Risk assessment
- Clinical recommendations
- Probability distribution

## Error Handling

- Image validation
- File type checking
- Size limitations
- Processing error management
- User-friendly error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers
- Open source contributors
- Research papers and references

## Contact

For support or queries, please contact [your-email@example.com] 
