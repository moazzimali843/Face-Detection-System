import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from datetime import datetime
import urllib.request
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. Data Preparation
def load_dataset(dataset_path):
    images = []
    labels = []
    
    person_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    for person in person_folders:
        person_path = os.path.join(dataset_path, person)
        for image_name in os.listdir(person_path):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_path, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (224, 224))
                    images.append(image)
                    labels.append(person)
    
    return np.array(images), np.array(labels)

# 2. Preprocess Data
def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return images, encoded_labels, label_encoder

# 3. Create CNN Model
def create_model(num_classes):
    # Use MobileNetV2 as base model (lightweight and effective)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 4. Detect and Recognize Faces in New Image (Using OpenCV DNN)
def detect_and_recognize_faces(image_path, model, label_encoder):
    # Load the image
    image = cv2.imread(image_path)
    
    # Define paths to the face detection model files
    # Update these paths to point to where you'll store the model files
    proto_path = "models/deploy.prototxt"
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(proto_path), exist_ok=True)
    
    # Check if model files exist, if not download them
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        print("Downloading face detection model files...")
        # Download prototxt
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        urllib.request.urlretrieve(prototxt_url, proto_path)
        
        # Download caffemodel
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        urllib.request.urlretrieve(model_url, model_path)
        print("Model files downloaded successfully!")
    
    # Load the face detection model
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    
    (h, w) = image.shape[:2]
    
    # Prepare the image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    results = []
    
    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Increase face detection confidence threshold
            # Get face coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (left, top, right, bottom) = box.astype("int")
            
            # Extract and preprocess face
            face_image = image[top:bottom, left:right]
            if face_image.size == 0:  # Check if face region is valid
                continue
            face_image = cv2.resize(face_image, (224, 224))
            face_image = face_image.astype('float32') / 255.0
            face_image = np.expand_dims(face_image, axis=0)
            
            # Predict
            prediction = model.predict(face_image, verbose=0)  # Add verbose=0 to reduce output
            person_idx = np.argmax(prediction)
            pred_confidence = np.max(prediction)
            
            # Get person name
            person_name = label_encoder.inverse_transform([person_idx])[0]
            
            # Add a dynamic confidence threshold based on number of classes
            min_confidence = 0.5 if len(label_encoder.classes_) < 5 else 0.7
            
            if pred_confidence > min_confidence:
                results.append({
                    'person': person_name,
                    'confidence': float(pred_confidence),
                    'location': (top, right, bottom, left)
                })
    
    return results

# 5. Main Execution
def main():
    dataset_path = 'dataset'
    
    print("Loading dataset...")
    images, labels = load_dataset(dataset_path)
    images, encoded_labels, label_encoder = preprocess_data(images, labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42
    )
    
    # Create data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    print("Training model...")
    model = create_model(num_classes=len(np.unique(labels)))
    
    # Create learning rate scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train with data augmentation and learning rate scheduling
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              steps_per_epoch=len(X_train) // 32,
              epochs=30,  # Increase max epochs, early stopping will prevent overfitting
              validation_data=(X_test, y_test),
              callbacks=[reduce_lr, early_stopping])
    
    # Fine-tuning
    model = fine_tune_model(model, X_train, y_train, X_test, y_test, datagen)
    
    # Save in both formats
    model.save(f'face_recognition_model_{datetime.now().strftime("%Y%m%d")}.keras')  # Modern format
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    
    return model, label_encoder

# 6. Test New Image
def test_new_image(image_path, model, label_encoder):
    results = detect_and_recognize_faces(image_path, model, label_encoder)
    
    image = cv2.imread(image_path)
    
    if not results:
        print("No faces detected in the image.")
        return
    
    print("\nDetection Results:")
    for i, result in enumerate(results, 1):
        person = result['person']
        confidence = result['confidence']
        top, right, bottom, left = result['location']
        
        print(f"Person {i}: {person} (Confidence: {confidence:.2%})")
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f"{person} ({confidence:.2%})", 
                   (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    output_path = f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
    cv2.imwrite(output_path, image)
    print(f"Output image saved as: {output_path}")

# Add this function after main()
def fine_tune_model(model, X_train, y_train, X_test, y_test, datagen):
    print("Fine-tuning the model...")
    
    # Unfreeze some layers
    for layer in model.layers[-20:]:  # Unfreeze the last 20 layers
        layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Create callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Fine-tune
    model.fit(datagen.flow(X_train, y_train, batch_size=16),
              steps_per_epoch=len(X_train) // 16,
              epochs=15,
              validation_data=(X_test, y_test),
              callbacks=[reduce_lr, early_stopping])
    
    return model

if __name__ == "__main__":
    # Train the model
    model, label_encoder = main()
    
    # Test with a new image - using a relative path in the project directory
    test_image_path = "test.jpg"  # Place a test image in your project root directory
    
    # Check if the test image exists
    if os.path.exists(test_image_path):
        test_new_image(test_image_path, model, label_encoder)
    else:
        print(f"Test image not found at {test_image_path}. Please add a test image or update the path.")
        # Prompt user for a valid path
        user_path = input("Enter the full path to a test image: ")
        if os.path.exists(user_path):
            test_new_image(user_path, model, label_encoder)
        else:
            print(f"Image not found at {user_path}. Exiting without testing.")