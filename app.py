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
import mediapipe as mp
import dlib
from imutils import face_utils
import zipfile
import shutil
import random

# Load environment variables from .env file
load_dotenv()

# Import functions from training.py
from training import  load_encodings

# Initialize MediaPipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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

# Install the shape predictor model if not already present
def download_shape_predictor():
    import urllib.request
    import bz2
    import os
    import ssl
    import certifi
    
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print("Downloading shape predictor model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = "shape_predictor_68_face_landmarks.dat.bz2"
        
        try:
            # Try with proper SSL context first
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, context=ssl_context) as response, open(compressed_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Downloaded with SSL verification")
        except Exception as e:
            print(f"SSL download failed: {str(e)}")
            try:
                # If that fails, try without SSL verification (not recommended but may work)
                print("Attempting download without SSL verification...")
                ssl_context = ssl._create_unverified_context()
                with urllib.request.urlopen(url, context=ssl_context) as response, open(compressed_path, 'wb') as out_file:
                    out_file.write(response.read())
                print("Downloaded without SSL verification")
            except Exception as e2:
                print(f"Non-SSL download failed: {str(e2)}")
                print("Using alternative approach...")
                # If both methods fail, use a simpler approach
                try:
                    import requests
                    print("Downloading with requests library...")
                    response = requests.get(url, verify=False)
                    with open(compressed_path, 'wb') as f:
                        f.write(response.content)
                    print("Downloaded with requests library")
                except Exception as e3:
                    print(f"Requests download failed: {str(e3)}")
                    raise Exception("All download methods failed")
        
        # Decompress the file
        try:
            print("Decompressing file...")
            with open(model_path, 'wb') as new_file, bz2.BZ2File(compressed_path, 'rb') as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)
            
            # Remove the compressed file
            os.remove(compressed_path)
            print("Shape predictor model downloaded and extracted.")
        except Exception as e:
            print(f"Decompression failed: {str(e)}")
            raise
    else:
        print(f"Shape predictor model already exists at {model_path}")
    
    return model_path

# Function to apply facial landmarks using dlib
def apply_face_mesh(image_path):
    print(f"Applying facial landmarks to image: {image_path}")
    
    try:
        # Try to get the shape predictor model
        try:
            model_path = download_shape_predictor()
            predictor = dlib.shape_predictor(model_path)
        except Exception as e:
            print(f"Error downloading shape predictor: {str(e)}")
            print("Using simplified face detection without landmarks...")
            # If we can't get the shape predictor, we'll just use face detection
            predictor = None
        
        # Initialize dlib's face detector
        detector = dlib.get_frontal_face_detector()
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        print("Detecting faces with dlib...")
        faces = detector(gray, 1)
        
        if len(faces) == 0:
            print("No faces detected with dlib")
            annotated_image = image.copy()
            cv2.putText(annotated_image, "No faces detected", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_image
        
        print(f"Detected {len(faces)} faces")
        
        # Create a copy of the image to draw on
        annotated_image = image.copy()
        
        # Process each face
        for i, face in enumerate(faces):
            print(f"Processing face #{i+1}")
            
            # Draw face bounding box
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # If we have the predictor, draw facial landmarks
            if predictor:
                # Get facial landmarks
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                
                # Draw facial landmarks
                for (x, y) in shape:
                    cv2.circle(annotated_image, (x, y), 2, (0, 0, 255), -1)
                
                # Draw connections between landmarks for different facial features
                # Eyes
                for start, end in [(36, 41), (42, 47)]:
                    points = shape[start:end+1]
                    for i in range(len(points) - 1):
                        cv2.line(annotated_image, tuple(points[i]), tuple(points[i+1]), (0, 255, 255), 1)
                    cv2.line(annotated_image, tuple(points[-1]), tuple(points[0]), (0, 255, 255), 1)
                
                # Eyebrows
                for start, end in [(17, 21), (22, 26)]:
                    points = shape[start:end+1]
                    for i in range(len(points) - 1):
                        cv2.line(annotated_image, tuple(points[i]), tuple(points[i+1]), (255, 0, 0), 1)
                
                # Nose
                points = shape[27:35+1]
                for i in range(len(points) - 1):
                    cv2.line(annotated_image, tuple(points[i]), tuple(points[i+1]), (0, 255, 0), 1)
                
                # Mouth
                points = shape[48:59+1]
                for i in range(len(points) - 1):
                    cv2.line(annotated_image, tuple(points[i]), tuple(points[i+1]), (255, 0, 255), 1)
                cv2.line(annotated_image, tuple(points[-1]), tuple(points[0]), (255, 0, 255), 1)
                
                # Inner mouth
                points = shape[60:67+1]
                for i in range(len(points) - 1):
                    cv2.line(annotated_image, tuple(points[i]), tuple(points[i+1]), (255, 0, 255), 1)
                cv2.line(annotated_image, tuple(points[-1]), tuple(points[0]), (255, 0, 255), 1)
            else:
                # If no predictor, just add a label to the face
                cv2.putText(annotated_image, f"Face #{i+1}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add a label to indicate this is the facial landmarks visualization
        if predictor:
            cv2.putText(annotated_image, "Facial Landmarks", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_image, "Face Detection Only", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        print("Face visualization completed successfully")
        return annotated_image
        
    except Exception as e:
        print(f"Error in face mesh processing: {str(e)}")
        print(traceback.format_exc())
        
        # Return a simple error image
        error_image = cv2.imread(image_path)
        if error_image is None:
            # Create a blank image if we can't read the original
            error_image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        cv2.putText(error_image, "Error processing face mesh", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_image, str(e), (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return error_image

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
                f"Describe the activities of each person in the image, focusing only on what they are doing individually or in a group. "
                f"Do not describe their physical appearance, clothing, or surroundings. Be concise and ensure that each person's actions are clearly stated. "
                f"The people in the image from LEFT to RIGHT are: {', '.join(ordered_names)}. "
                f"Please refer to them by their names and accurately describe their actions based on their position from left to right."
            )
        else:
            prompt_text = "Describe the activities of the people in the image, focusing only on what they are doing. Do not include details about their physical appearance, clothing, or the setting. Be concise and clear in describing their actions."
        
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

# Function to augment an image
def augment_image(image):
    """Applies a series of random augmentations to the image."""
    # Horizontal Flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Rotation
    angle = random.randint(-30, 30)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    image = cv2.warpAffine(image, matrix, (w, h))

    # Scaling
    scale = random.uniform(0.8, 1.2)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Cropping (random 80-100% of image)
    h, w = image.shape[:2]  # Get new dimensions after scaling
    crop_x = random.randint(0, max(1, w//5))
    crop_y = random.randint(0, max(1, h//5))
    if crop_y < h and crop_x < w:  # Ensure valid crop dimensions
        image = image[crop_y:h-crop_y, crop_x:w-crop_x]

    # Brightness/Contrast
    alpha = random.uniform(0.7, 1.3)  # Contrast
    beta = random.randint(-30, 30)  # Brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return image

# Function to train the face recognition model
def train_face_recognition_model(dataset_path):
    """Train the face recognition model and save encodings."""
    from training import train_and_save_encodings
    
    print(f"Training face recognition model with data from {dataset_path}")
    print(f"Directory contents: {os.listdir(dataset_path)}")
    
    # Check if the directory structure is correct
    person_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    print(f"Found {len(person_folders)} person folders: {person_folders}")
    
    # Check each person folder for images
    for person in person_folders:
        person_dir = os.path.join(dataset_path, person)
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Person '{person}' has {len(image_files)} images")
    
    # Train the model
    known_face_encodings, known_face_names = train_and_save_encodings(dataset_path, "known_faces.pkl")
    print(f"Training complete. Encoded {len(known_face_encodings)} faces for {len(set(known_face_names))} people.")
    
    return len(known_face_encodings), len(set(known_face_names))

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
            
            # Process the image using recognize_faces from training.py
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
            
            # Save the output image with face recognition results
            output_filename = f"output_{timestamp}.jpg"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, image)
            
            # Apply facial landmarks to the image
            print(f"Starting facial landmarks processing for {file_path}")
            mesh_image = apply_face_mesh(file_path)
            if mesh_image is not None:
                # Save the mesh image
                mesh_filename = f"mesh_{timestamp}.jpg"
                mesh_path = os.path.join(app.config['OUTPUT_FOLDER'], mesh_filename)
                print(f"Saving mesh image to {mesh_path}")
                cv2.imwrite(mesh_path, mesh_image)
                output_data['mesh_image_path'] = f"/outputs/{mesh_filename}"
                print(f"Mesh image path added to response: {output_data['mesh_image_path']}")
            else:
                print("Failed to generate mesh image")
            
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

@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'GET':
        return render_template('training.html')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.zip'):
            try:
                # Create temporary directories
                temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_training')
                dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset')
                augmented_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'augmented_dataset')
                
                # Clean up any existing directories
                for dir_path in [temp_dir, dataset_dir, augmented_dir]:
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
                    os.makedirs(dir_path)
                
                # Save and extract the zip file
                zip_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(zip_path)
                
                print(f"Saved ZIP file to {zip_path}")
                print(f"Extracting zip file to {dataset_dir}")

                # Check the ZIP file contents before extraction
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    print(f"ZIP contains {len(file_list)} files/folders")
                    print(f"First 10 entries: {file_list[:10]}")
                    
                    # Extract the ZIP file
                    zip_ref.extractall(dataset_dir)

                print(f"Extraction complete. Dataset directory contents: {os.listdir(dataset_dir)}")

                # Check if we need to handle a nested directory structure
                # Sometimes ZIP files contain a single root folder
                if len(os.listdir(dataset_dir)) == 1:
                    first_item = os.path.join(dataset_dir, os.listdir(dataset_dir)[0])
                    if os.path.isdir(first_item):
                        print(f"Found single root directory: {first_item}")
                        # If the ZIP contained a single folder, use its contents as our dataset
                        if len(os.listdir(first_item)) > 0:
                            print(f"Moving contents from {first_item} to {dataset_dir}")
                            for item in os.listdir(first_item):
                                shutil.move(
                                    os.path.join(first_item, item),
                                    os.path.join(dataset_dir, item)
                                )
                            # Remove the now-empty directory
                            os.rmdir(first_item)
                            print(f"After restructuring, dataset directory contains: {os.listdir(dataset_dir)}")
                
                # Process each person's folder
                person_count = 0
                total_images = 0
                augmented_images = 0
                
                print(f"Dataset directory contents: {os.listdir(dataset_dir)}")

                for person_folder in os.listdir(dataset_dir):
                    folder_path = os.path.join(dataset_dir, person_folder)
                    
                    if not os.path.isdir(folder_path):
                        print(f"Skipping {person_folder} as it's not a directory")
                        continue  # Skip if not a directory
                    
                    person_count += 1
                    output_folder_path = os.path.join(augmented_dir, person_folder)
                    os.makedirs(output_folder_path, exist_ok=True)
                    
                    # Copy original images to augmented directory
                    images = []
                    for img_name in os.listdir(folder_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(folder_path, img_name)
                            new_img_path = os.path.join(output_folder_path, img_name)
                            print(f"Copying {img_path} to {new_img_path}")
                            shutil.copy2(img_path, new_img_path)
                            images.append(img_name)
                    
                    original_count = len(images)
                    total_images += original_count
                    
                    print(f"Processing {person_folder}: {original_count} original images")
                    
                    # If no images were found, print a warning
                    if original_count == 0:
                        print(f"WARNING: No images found for {person_folder}!")
                        print(f"Folder contents: {os.listdir(folder_path)}")
                        continue
                    
                    # Apply augmentation to create additional images
                    count = original_count + 1
                    target_count = min(500, max(50, original_count * 10))  # Aim for 10x but cap at 500
                    
                    print(f"Augmenting images for {person_folder} from {original_count} to target {target_count}")
                    
                    while len(images) < target_count and original_count > 0:
                        # Pick a random image to augment
                        img_name = random.choice(images[:original_count])  # Only choose from original images
                        img_path = os.path.join(folder_path, img_name)
                        
                        # Load image
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Error loading {img_path}, skipping...")
                            continue
                        
                        # Apply augmentation
                        try:
                            augmented_img = augment_image(img)
                            
                            # Save new image in the output folder
                            new_img_name = f"{person_folder}_{count}.png"
                            new_img_path = os.path.join(output_folder_path, new_img_name)
                            cv2.imwrite(new_img_path, augmented_img)
                            
                            images.append(new_img_name)  # Add new image to list
                            count += 1
                            augmented_images += 1
                        except Exception as e:
                            print(f"Error augmenting image: {str(e)}")
                    
                    print(f"Augmentation complete for {person_folder}, now contains {len(images)} images")
                
                # Train the model using the augmented dataset
                face_count, person_count = train_face_recognition_model(augmented_dir)
                
                # Clean up
                for dir_path in [temp_dir, dataset_dir, augmented_dir]:
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
                
                return jsonify({
                    'message': 'Training completed successfully',
                    'stats': {
                        'persons': person_count,
                        'original_images': total_images,
                        'augmented_images': augmented_images,
                        'encoded_faces': face_count
                    }
                })
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                print(traceback.format_exc())
                
                # Clean up on error
                for dir_path in [temp_dir, dataset_dir, augmented_dir]:
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
                
                return jsonify({'error': f'An error occurred during training: {str(e)}'}), 500
        
        return jsonify({'error': 'Invalid file type. Please upload a ZIP file.'}), 400

@app.route('/testing', methods=['GET'])
def testing():
    return render_template('testing.html')

if __name__ == '__main__':
    import face_recognition  # Import here to ensure it's available
    app.run(debug=True) 