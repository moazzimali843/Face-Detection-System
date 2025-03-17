import cv2
import numpy as np
import face_recognition
import os
import pickle
import base64
import openai

def get_image_description(image_path, detected_persons, api_key):
    """Get image description using OpenAI Vision with detected names and positions."""
    try:
        # Initialize OpenAI client with API key
        client = openai.OpenAI(api_key=api_key)
        
        # Read image and convert to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Extract names from detected persons (already sorted from left to right)
        ordered_names = [person['name'] for person in detected_persons if person['name'] != "Unknown"]
        
        # Create a prompt based on detected persons
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
            model="gpt-4-vision-preview",
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
        from modules.utils import log_exception
        log_exception(e, "Error getting image description")
        return "Could not generate image description."

def recognize_faces_in_image(image_path, encodings_file="known_faces.pkl"):
    """Recognize faces in an image using the trained model."""
    try:
        # Load the trained model
        if not os.path.exists(encodings_file):
            raise FileNotFoundError(f"Face encodings not found at {encodings_file}. Please train the model first.")
        
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
        
        known_face_encodings = data['encodings']
        known_face_names = data['names']
        
        print(f"Loaded {len(known_face_encodings)} face encodings from {encodings_file}")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        print(f"Detected {len(face_locations)} faces in the image")
        
        # Process each face
        persons = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Calculate face distances
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            # Find the best match
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                distance = face_distances[best_match_index]
                
                # Set a threshold for recognition
                if distance < 0.6:
                    name = known_face_names[best_match_index]
                    confidence = 1.0 - distance  # Convert distance to confidence
                else:
                    name = "Unknown"
                    confidence = 0.0
            else:
                name = "Unknown"
                confidence = 0.0
            
            # Draw rectangle and name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, f"{name} ({confidence:.2f})", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Add to persons list
            persons.append({
                'name': name,
                'confidence': float(confidence),
                'box': [left, top, right, bottom],
                'x': left  # For sorting left to right
            })
        
        # Sort persons from left to right
        persons.sort(key=lambda p: p['x'])
        
        return image, persons
        
    except Exception as e:
        from modules.utils import log_exception
        log_exception(e, "Error recognizing faces")
        raise 