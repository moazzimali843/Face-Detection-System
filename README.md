# FACE DETECTION SYSTEM
A simple face recognition system with training and testing modules, allowing users to train on a custom dataset and recognize faces in images.

## Installation & Setup
Clone the repository:
```
git clone https://github.com/moazzimali843/Face-Detection-System
cd Face-Detection-System
```

## Create a virtual environment:
```
python -m venv myenv
```
## Activate the virtual environment:
On Windows:
```
myenv\Scripts\activate
```
On macOS/Linux:
```
source myenv/bin/activate
```
## Install dependencies:
```
pip install -r requirements.txt
```
## Create a .env file and add your OpenAI API key:
```
OPENAI_API_KEY="sk-proj-1234567890" 
```
## Important Point:
- Delete the file known_faces.pkl before starting the training process.
- Training will create a new known_faces.pkl file with the new encodings.

## Run the application:
```
python app.py
```

# Modules Overview
## 1. Training Module
- Allows users to upload and train the model on a custom dataset.
- Processes images by applying augmentation to increase dataset size.
- Discards images where faces are not properly detected.
- Training takes time, and once completed, the model is saved.
## 2. Testing & Recognition Module
- Users can upload a single image or a group photo of trained individuals.
- Performs the following tasks:
    - Face detection and drawing bounding boxes around faces.
    - Identifies & names recognized individuals.
    - Applies a facial mesh for better visualization.
    - Describes the scene (e.g., actions, expressions).

## Image Upload Format:
- Upload a .zip file containing multiple folders.
- Each folder represents one person and should contain their individual images.
- Folder names should match the person's name for accurate training.
- More & clearer photos → Better results!

# License
This project is open-source. Feel free to modify and improve it! 🚀
