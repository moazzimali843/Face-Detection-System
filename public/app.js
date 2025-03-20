// Firebase configuration
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "face-recognition-app.firebaseapp.com",
  projectId: "face-recognition-app",
  storageBucket: "face-recognition-app.appspot.com",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Get references to Firebase services
const auth = firebase.auth();
const storage = firebase.storage();
const functions = firebase.functions();

// DOM elements
const loginBtn = document.getElementById('login-btn');
const authContainer = document.getElementById('auth-container');
const appContainer = document.getElementById('app-container');
const imageUpload = document.getElementById('image-upload');
const uploadBtn = document.getElementById('upload-btn');
const newFaceUpload = document.getElementById('new-face-upload');
const faceName = document.getElementById('face-name');
const addFaceBtn = document.getElementById('add-face-btn');
const resultContainer = document.getElementById('result-container');
const previewImage = document.getElementById('preview-image');
const resultName = document.getElementById('result-name');
const resultDescription = document.getElementById('result-description');

// Authentication state observer
auth.onAuthStateChanged(user => {
  if (user) {
    // User is signed in
    authContainer.classList.add('hidden');
    appContainer.classList.remove('hidden');
  } else {
    // User is signed out
    authContainer.classList.remove('hidden');
    appContainer.classList.add('hidden');
    resultContainer.classList.add('hidden');
  }
});

// Login with Google
loginBtn.addEventListener('click', () => {
  const provider = new firebase.auth.GoogleAuthProvider();
  auth.signInWithPopup(provider).catch(error => {
    console.error('Error signing in:', error);
    alert(`Error signing in: ${error.message}`);
  });
});

// Upload and recognize face
uploadBtn.addEventListener('click', async () => {
  if (!imageUpload.files.length) {
    alert('Please select an image first');
    return;
  }

  try {
    // Show loading state
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Processing...';
    
    // Upload image to Firebase Storage
    const file = imageUpload.files[0];
    const storageRef = storage.ref(`uploads/${Date.now()}_${file.name}`);
    await storageRef.put(file);
    
    // Get download URL
    const imageUrl = await storageRef.getDownloadURL();
    
    // Call the Cloud Function
    const processFaceRecognition = functions.httpsCallable('processFaceRecognition');
    const result = await processFaceRecognition({ imageUrl });
    
    // Display the result
    previewImage.src = imageUrl;
    
    if (result.data.matched) {
      resultName.textContent = result.data.name;
      resultDescription.textContent = result.data.description;
    } else {
      resultName.textContent = 'No Match Found';
      resultDescription.textContent = 'This person was not found in the database.';
    }
    
    resultContainer.classList.remove('hidden');
  } catch (error) {
    console.error('Error processing image:', error);
    alert(`Error: ${error.message}`);
  } finally {
    // Reset button state
    uploadBtn.disabled = false;
    uploadBtn.textContent = 'Recognize Face';
  }
});

// Add new face to database
addFaceBtn.addEventListener('click', async () => {
  if (!newFaceUpload.files.length) {
    alert('Please select an image first');
    return;
  }
  
  if (!faceName.value.trim()) {
    alert('Please enter a name');
    return;
  }

  try {
    // Show loading state
    addFaceBtn.disabled = true;
    addFaceBtn.textContent = 'Adding...';
    
    // Upload image to Firebase Storage
    const file = newFaceUpload.files[0];
    const storageRef = storage.ref(`known_faces/${Date.now()}_${file.name}`);
    await storageRef.put(file);
    
    // Get download URL
    const imageUrl = await storageRef.getDownloadURL();
    
    // Call the Cloud Function
    const addFaceToDatabase = functions.httpsCallable('addFaceToDatabase');
    await addFaceToDatabase({ 
      imageUrl, 
      name: faceName.value.trim() 
    });
    
    // Reset form
    newFaceUpload.value = '';
    faceName.value = '';
    
    alert('Face added successfully!');
  } catch (error) {
    console.error('Error adding face:', error);
    alert(`Error: ${error.message}`);
  } finally {
    // Reset button state
    addFaceBtn.disabled = false;
    addFaceBtn.textContent = 'Add to Database';
  }
}); 