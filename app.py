import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from datetime import datetime
import json
import gdown

# Import functions from training.py
from training import detect_and_recognize_faces

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the trained model and label encoder
def load_recognition_model():
    # Find the most recent model file
    model_files = [f for f in os.listdir('.') if f.endswith('.keras') and f.startswith('face_recognition_model_')]
    if not model_files:
        return None, None
    
    latest_model = max(model_files)
    model = load_model(latest_model)
    
    # Load label encoder classes
    if os.path.exists('label_encoder_classes.npy'):
        label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
        
        # Create a simple label encoder-like object
        class SimpleEncoder:
            def __init__(self, classes):
                self.classes_ = classes
                
            def inverse_transform(self, indices):
                return [self.classes_[i] for i in indices]
        
        label_encoder = SimpleEncoder(label_encoder_classes)
        return model, label_encoder
    
    return model, None

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Load model and process image
        model, label_encoder = load_recognition_model()
        
        if model is None or label_encoder is None:
            return jsonify({'error': 'Model or label encoder not found. Train the model first.'}), 500
        
        # Process the image
        results = detect_and_recognize_faces(file_path, model, label_encoder)
        
        # Create output image with annotations
        image = cv2.imread(file_path)
        
        if not results:
            output_data = {'message': 'No faces detected in the image.', 'persons': []}
        else:
            persons = []
            for i, result in enumerate(results, 1):
                person = result['person']
                confidence = result['confidence']
                top, right, bottom, left = result['location']
                
                persons.append({
                    'name': person,
                    'confidence': float(confidence)
                })
                
                # Draw rectangle and text on the image
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image, f"{person} ({confidence:.2%})", 
                           (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            output_data = {'message': f'Detected {len(results)} face(s)', 'persons': persons}
        
        # Save the output image
        output_filename = f"output_{timestamp}.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, image)
        
        # Add the output image path to the response
        output_data['image_path'] = f"/outputs/{output_filename}"
        
        return jsonify(output_data)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def download_model_if_needed():
    model_files = [f for f in os.listdir('.') if f.endswith('.keras') and f.startswith('face_recognition_model_')]
    if not model_files:
        print("Downloading model from external source...")
        # Example using gdown for Google Drive
        url = 'YOUR_GOOGLE_DRIVE_LINK'
        output = 'face_recognition_model_latest.keras'
        gdown.download(url, output, quiet=False)
        
        # Also download label encoder classes if needed
        if not os.path.exists('label_encoder_classes.npy'):
            url_labels = 'YOUR_GOOGLE_DRIVE_LINK_FOR_LABELS'
            gdown.download(url_labels, 'label_encoder_classes.npy', quiet=False)

# Call this function before loading the model
download_model_if_needed()

if __name__ == '__main__':
    app.run(debug=True) 