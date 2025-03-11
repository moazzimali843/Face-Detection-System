import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

def detect_and_recognize_faces(image_path, model, label_encoder):
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path: {image_path}")
        return []
        
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at path: {image_path}")
        return []
        
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces using face_recognition library
    face_locations = face_recognition.face_locations(rgb_image)
    
    results = []
    
    for face_location in face_locations:
        top, right, bottom, left = face_location
        
        # Extract face
        face_image = image[top:bottom, left:right]
        face_image = cv2.resize(face_image, (224, 224))
        face_image = face_image.astype('float32') / 255.0
        face_image = np.expand_dims(face_image, axis=0)
        
        # Predict
        prediction = model.predict(face_image)
        person_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Get person name
        person_name = label_encoder.inverse_transform([person_idx])[0]
        
        if confidence > 0.7:  # Confidence threshold
            results.append({
                'person': person_name,
                'confidence': float(confidence),
                'location': (top, right, bottom, left)
            })
    
    return results

def test_new_image(image_path, model, label_encoder):
    results = detect_and_recognize_faces(image_path, model, label_encoder)
    
    if not results:
        print("No faces detected in the image or image could not be processed.")
        return
    
    # Load image for visualization
    image = cv2.imread(image_path)
    
    print("\nDetection Results:")
    for i, result in enumerate(results, 1):
        person = result['person']
        confidence = result['confidence']
        top, right, bottom, left = result['location']
        
        print(f"Person {i}: {person} (Confidence: {confidence:.2%})")
        
        # Draw rectangle and label on image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f"{person} ({confidence:.2%})", 
                   (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save output image
    output_path = f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
    cv2.imwrite(output_path, image)
    print(f"Output image saved as: {output_path}")

def load_trained_model():
    # Find the most recent model file
    model_files = [f for f in os.listdir('.') if f.startswith('face_recognition_model_') and f.endswith('.h5')]
    
    if not model_files:
        print("Error: No trained model found. Please train the model first.")
        return None, None
    
    # Sort by date (newest first)
    model_files.sort(reverse=True)
    model_path = model_files[0]
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load label encoder classes
    if os.path.exists('label_encoder_classes.npy'):
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)
        return model, label_encoder
    else:
        print("Error: Label encoder classes not found.")
        return model, None

if __name__ == "__main__":
    # Load the trained model
    model, label_encoder = load_trained_model()
    
    if model is None or label_encoder is None:
        print("Could not load model or label encoder. Exiting.")
        exit(1)
    
    # Ask user for image path
    print("\nEnter the path to the image you want to test:")
    print("(You can use a relative path like 'test.jpg' or an absolute path)")
    
    image_path = input("Image path: ").strip()
    
    if os.path.exists(image_path):
        test_new_image(image_path, model, label_encoder)
    else:
        print(f"Image not found at {image_path}.")
        print("Please check the path and try again.") 