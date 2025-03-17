import face_recognition
import cv2
import numpy as np
import os
import pickle

def train_and_save_encodings(dataset_path, encodings_file="known_faces.pkl"):
    """Train the face recognition model and save encodings to file."""
    known_face_encodings = []
    known_face_names = []

    print(f"Training face recognition model with data from {dataset_path}")
    print(f"Directory contents: {os.listdir(dataset_path)}")
    
    # Check if the directory structure is correct
    person_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    print(f"Found {len(person_folders)} person folders: {person_folders}")
    
    # Process each person's folder
    for person_name in person_folders:
        person_folder = os.path.join(dataset_path, person_name)
        
        # Check each person folder for images
        image_files = [f for f in os.listdir(person_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Person '{person_name}' has {len(image_files)} images")
        
        # Process each image
        for image_name in image_files:
            image_path = os.path.join(person_folder, image_name)
            try:
                # Load image and convert to RGB
                image = face_recognition.load_image_file(image_path)
                # Detect faces and get encodings
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)
                    print(f"Encoded {image_name} for {person_name}")
                else:
                    print(f"No face found in {image_name}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
    
    # Save encodings and names to a file
    with open(encodings_file, 'wb') as f:
        pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, f)
    print(f"Saved {len(known_face_encodings)} face encodings to {encodings_file}")
    
    return known_face_encodings, known_face_names

def load_encodings(encodings_file="known_faces.pkl"):
    """Load precomputed encodings from file."""
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['encodings'])} face encodings from {encodings_file}")
        return data['encodings'], data['names']
    else:
        raise FileNotFoundError(f"Encodings file {encodings_file} not found. Please train the model first.")

def train_face_recognition_model(dataset_path):
    """Train the face recognition model and return statistics."""
    known_face_encodings, known_face_names = train_and_save_encodings(dataset_path, "known_faces.pkl")
    print(f"Training complete. Encoded {len(known_face_encodings)} faces for {len(set(known_face_names))} people.")
    
    return len(known_face_encodings), len(set(known_face_names)) 