import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import pickle
import traceback
import base64
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import functions from training2.py instead of training.py
from training2 import recognize_faces, load_encodings

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize OpenAI client with API key from environment variable
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to get image description using OpenAI Vision with detected names and positions
def get_image_description(image_path, detected_persons):
    try:
        # Read image and convert to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Extract names from detected persons (already sorted from left to right)
        ordered_names = [person['name'] for person in detected_persons if person['name'] != "Unknown"]
        
        # Create a prompt that includes the detected names with positions
        if ordered_names:
            # Create a descriptive prompt with positional information
            prompt_text = (
                f"Describe what these people are doing in this image. Focus on their activities, clothing, and the setting. "
                f"The people in the image from LEFT to RIGHT are: {', '.join(ordered_names)}. "
                f"Please refer to them by their names and ensure you match the correct name to the correct person based on their position from left to right. "
                f"Be concise and descriptive."
            )
        else:
            prompt_text = "Describe what these people are doing in this image. Focus on their activities, clothing, and the setting. Be concise."
        
        # Call OpenAI API with the correct format for image_url
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting image description: {str(e)}")
        return "Could not generate image description."

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
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Load face encodings
            try:
                known_face_encodings, known_face_names = load_encodings("known_faces.pkl")
            except FileNotFoundError:
                return jsonify({'error': 'Face encodings not found. Please train the model first.'}), 500
            
            # Process the image using recognize_faces from training2.py
            # We'll use the original image for processing
            image = cv2.imread(file_path)
            if image is None:
                return jsonify({'error': 'Failed to read the uploaded image'}), 500
                
            # Convert image to RGB for face_recognition library
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find all face locations and encodings in the test image
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            # Process the results directly here instead of using recognize_faces
            if len(face_encodings) == 0:
                output_data = {'message': 'No faces detected in the image.', 'persons': []}
            else:
                persons = []
                
                # Sort face locations from left to right
                # This ensures we process faces in a consistent order
                sorted_faces = sorted(zip(face_locations, face_encodings), key=lambda x: x[0][3])  # Sort by left coordinate
                face_locations = [loc for loc, _ in sorted_faces]
                face_encodings = [enc for _, enc in sorted_faces]
                
                # Process each detected face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Calculate distances between the unknown face and all known faces
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    # Find the best match (smallest distance)
                    best_match_index = np.argmin(face_distances)
                    distance = face_distances[best_match_index]
                    
                    # Set a threshold for recognition
                    if distance < 0.6:
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                    
                    # Calculate confidence (inverse of distance)
                    confidence = 1.0 - distance
                    
                    # Add location information to the person data
                    persons.append({
                        'name': name,
                        'confidence': float(confidence),
                        'location': (left, top, right, bottom)  # Store location for sorting
                    })
                    
                    # Draw rectangle and text on the image
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(image, f"{name} ({confidence:.2%})", 
                               (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Sort persons from left to right based on the left coordinate
                persons = sorted(persons, key=lambda x: x['location'][0])
                
                output_data = {'message': f'Detected {len(persons)} face(s)', 'persons': persons}
            
            # Save the output image
            output_filename = f"output_{timestamp}.jpg"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, image)
            
            # Get image description using OpenAI Vision with detected persons and face locations
            image_description = get_image_description(file_path, persons)
            
            # Add the output image path and description to the response
            output_data['image_path'] = f"/outputs/{output_filename}"
            output_data['image_description'] = image_description
            
            return jsonify(output_data)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'An error occurred while processing the image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    import face_recognition  # Import here to ensure it's available
    app.run(debug=True) 