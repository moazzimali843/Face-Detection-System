import cv2
import dlib
from imutils import face_utils
import numpy as np
from modules.utils import download_shape_predictor

def apply_face_mesh(image_path):
    """Apply facial landmarks using dlib."""
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
        from modules.utils import log_exception
        log_exception(e, "Error in face mesh processing")
        
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