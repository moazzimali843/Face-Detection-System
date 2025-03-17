import os
import urllib.request
import bz2
import ssl
import certifi
import traceback

def ensure_dir_exists(directory):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    
def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def download_shape_predictor():
    """Download the shape predictor model if not already present."""
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

def log_exception(e, message="An error occurred"):
    """Log an exception with traceback."""
    print(f"{message}: {str(e)}")
    print(traceback.format_exc()) 