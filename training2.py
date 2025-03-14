# import face_recognition
# import cv2
# import numpy as np
# import os
# from PIL import Image

# # Step 1: Load and encode faces from the dataset
# def load_and_encode_faces(dataset_path):
#     known_face_encodings = []
#     known_face_names = []

#     # Loop through each person's folder
#     for person_name in os.listdir(dataset_path):
#         person_folder = os.path.join(dataset_path, person_name)
#         if os.path.isdir(person_folder):
#             for image_name in os.listdir(person_folder):
#                 image_path = os.path.join(person_folder, image_name)
#                 try:
#                     # Load image and convert to RGB
#                     image = face_recognition.load_image_file(image_path)
#                     # Detect faces and get encodings
#                     face_encodings = face_recognition.face_encodings(image)
#                     if len(face_encodings) > 0:
#                         known_face_encodings.append(face_encodings[0])
#                         known_face_names.append(person_name)
#                         print(f"Encoded {image_name} for {person_name}")
#                     else:
#                         print(f"No face found in {image_name}")
#                 except Exception as e:
#                     print(f"Error processing {image_name}: {e}")
    
#     return known_face_encodings, known_face_names

# # Step 2: Recognize all faces in a group photo
# def recognize_faces(image_path, known_face_encodings, known_face_names):
#     # Load the test image
#     unknown_image = face_recognition.load_image_file(image_path)
    
#     # Find all face locations and encodings in the test image
#     face_locations = face_recognition.face_locations(unknown_image)
#     face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
#     # If no faces are found
#     if len(face_encodings) == 0:
#         return "No faces detected in the image."
    
#     # Convert to BGR for OpenCV visualization
#     cv_image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)
#     results = []
    
#     # Process each detected face
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Calculate distances between the unknown face and all known faces
#         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
#         # Find the best match (smallest distance)
#         best_match_index = np.argmin(face_distances)
#         distance = face_distances[best_match_index]
        
#         # Set a threshold for recognition
#         if distance < 0.6:
#             name = known_face_names[best_match_index]
#         else:
#             name = "Unknown"
        
#         # Draw a box and label on the image
#         cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(cv_image, f"{name} ({distance:.2f})", (left, top - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         # Store the result
#         results.append(f"Person at ({left}, {top}, {right}, {bottom}): {name} (distance: {distance:.2f})")
    
#     # Show the result image with all detections
#     cv2.imshow("Group Photo Result", cv_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return "\n".join(results) if results else "No recognizable faces found."

# # Main execution
# if __name__ == "__main__":
#     # Path to your dataset
#     dataset_path = "final_dataset/"
    
#     # Step 1: Train the model by encoding known faces
#     print("Loading and encoding faces from dataset...")
#     known_face_encodings, known_face_names = load_and_encode_faces(dataset_path)
#     print(f"Loaded {len(known_face_encodings)} face encodings.")
    
#     # Step 2: Test with a group photo
#     test_image_path = "test2.jpg"  # Replace with your group photo path
#     print("\nRecognizing faces in group photo...")
#     result = recognize_faces(test_image_path, known_face_encodings, known_face_names)
#     print(result)



import face_recognition
import cv2
import numpy as np
import os
import pickle

# Step 1: Load and encode faces from the dataset and save to file
def train_and_save_encodings(dataset_path, encodings_file="known_faces.pkl"):
    known_face_encodings = []
    known_face_names = []

    # Loop through each person's folder
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
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

# Step 2: Load precomputed encodings from file
def load_encodings(encodings_file="known_faces.pkl"):
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['encodings'])} face encodings from {encodings_file}")
        return data['encodings'], data['names']
    else:
        raise FileNotFoundError(f"Encodings file {encodings_file} not found. Please train the model first.")

# Step 3: Recognize all faces in a group photo using precomputed encodings
def recognize_faces(image_path, known_face_encodings, known_face_names, display_window=True):
    # Load the test image
    unknown_image = face_recognition.load_image_file(image_path)
    
    # Find all face locations and encodings in the test image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    # If no faces are found
    if len(face_encodings) == 0:
        return "No faces detected in the image."
    
    # Convert to BGR for OpenCV visualization
    cv_image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)
    results = []
    
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
        
        # Draw a box and label on the image
        cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(cv_image, f"{name} ({distance:.2f})", (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Store the result
        results.append(f"Person at ({left}, {top}, {right}, {bottom}): {name} (distance: {distance:.2f})")
    
    # Only show the OpenCV window if display_window is True (for command line usage)
    # When called from the web app, this should be False
    if display_window:
        cv2.imshow("Group Photo Result", cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save the processed image to a temporary file that can be returned
    output_path = f"temp_output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, cv_image)
    
    return "\n".join(results) if results else "No recognizable faces found."

# Main execution
if __name__ == "__main__":
    dataset_path = "final_dataset/"  # Path to your dataset
    encodings_file = "known_faces.pkl"  # File to store encodings
    test_image_path = "test3.jpg"  # Replace with your test image path
    
    # Check if encodings file exists
    if os.path.exists(encodings_file):
        # Load precomputed encodings
        print("Loading precomputed encodings...")
        known_face_encodings, known_face_names = load_encodings(encodings_file)
    else:
        # Train the model and save encodings
        print("Training model and saving encodings...")
        known_face_encodings, known_face_names = train_and_save_encodings(dataset_path, encodings_file)
    
    # Recognize faces in the test image
    print("\nRecognizing faces in test image...")
    # When running directly, we want to display the window
    result = recognize_faces(test_image_path, known_face_encodings, known_face_names, display_window=True)
    print(result)