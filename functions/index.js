const functions = require('firebase-functions');
const admin = require('firebase-admin');
const { OpenAI } = require('openai');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const cors = require('cors')({ origin: true });

admin.initializeApp();

// Load the TensorFlow Lite model
let model;
async function loadModel() {
  // Path to your TFLite model in Firebase Storage
  const modelPath = 'gs://face-recognition-app.appspot.com/models/face_recognition_model.tflite';
  model = await tf.loadGraphModel(modelPath);
  console.log('Face recognition model loaded');
}

// Initialize the model when the function starts
loadModel().catch(err => console.error('Failed to load model:', err));

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: functions.config().openai.key
});

exports.processFaceRecognition = functions.https.onRequest((req, res) => {
  cors(req, res, async () => {
    try {
      if (req.method !== 'POST') {
        return res.status(405).send('Method Not Allowed');
      }

      // Get the image URL from the request
      const { imageUrl } = req.body;
      if (!imageUrl) {
        return res.status(400).send('No image URL provided');
      }

      // Download the image from Firebase Storage
      const bucket = admin.storage().bucket();
      const tempFilePath = '/tmp/image.jpg';
      
      // Extract the file path from the URL
      const filePath = imageUrl.split('gs://face-recognition-app.appspot.com/')[1];
      await bucket.file(filePath).download({ destination: tempFilePath });

      // Process the image
      const imageBuffer = await sharp(tempFilePath)
        .resize(224, 224) // Resize to model input size
        .toBuffer();

      // Convert image to tensor
      const tensor = tf.node.decodeImage(imageBuffer);
      const expandedTensor = tensor.expandDims(0);
      const normalizedTensor = expandedTensor.div(255.0);

      // Run face detection
      const predictions = await model.predict(normalizedTensor);
      
      // Process predictions to extract face embeddings
      const faceEmbeddings = predictions.arraySync()[0];
      
      // Compare with known faces in database
      const matchedFace = await compareFaceWithDatabase(faceEmbeddings);
      
      // Generate description using OpenAI
      let description = '';
      if (matchedFace) {
        description = await generateDescription(matchedFace.name);
      } else {
        description = "No matching face found in the database.";
      }

      // Clean up
      tensor.dispose();
      expandedTensor.dispose();
      normalizedTensor.dispose();
      predictions.dispose();

      // Return the result
      res.status(200).json({
        matched: !!matchedFace,
        name: matchedFace ? matchedFace.name : null,
        description
      });
    } catch (error) {
      console.error('Error processing image:', error);
      res.status(500).send(`Error processing image: ${error.message}`);
    }
  });
});

// Function to compare face with database
async function compareFaceWithDatabase(faceEmbedding) {
  const snapshot = await admin.firestore().collection('knownFaces').get();
  
  let bestMatch = null;
  let highestSimilarity = 0;
  
  snapshot.forEach(doc => {
    const knownFace = doc.data();
    const similarity = cosineSimilarity(faceEmbedding, knownFace.embedding);
    
    if (similarity > 0.7 && similarity > highestSimilarity) {
      highestSimilarity = similarity;
      bestMatch = {
        id: doc.id,
        name: knownFace.name,
        similarity
      };
    }
  });
  
  return bestMatch;
}

// Calculate cosine similarity between two embeddings
function cosineSimilarity(embedding1, embedding2) {
  const dotProduct = embedding1.reduce((sum, value, i) => sum + value * embedding2[i], 0);
  const magnitude1 = Math.sqrt(embedding1.reduce((sum, value) => sum + value * value, 0));
  const magnitude2 = Math.sqrt(embedding2.reduce((sum, value) => sum + value * value, 0));
  
  return dotProduct / (magnitude1 * magnitude2);
}

// Generate description using OpenAI
async function generateDescription(name) {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant that generates brief descriptions of people."
        },
        {
          role: "user",
          content: `Generate a brief description for a person named ${name}.`
        }
      ],
      max_tokens: 150
    });
    
    return response.choices[0].message.content;
  } catch (error) {
    console.error('Error generating description:', error);
    return `This is ${name}.`;
  }
}

// Function to add a new face to the database
exports.addFaceToDatabase = functions.https.onRequest((req, res) => {
  cors(req, res, async () => {
    try {
      if (req.method !== 'POST') {
        return res.status(405).send('Method Not Allowed');
      }

      const { imageUrl, name } = req.body;
      if (!imageUrl || !name) {
        return res.status(400).send('Image URL and name are required');
      }

      // Download and process the image
      const bucket = admin.storage().bucket();
      const tempFilePath = '/tmp/new_face.jpg';
      
      const filePath = imageUrl.split('gs://face-recognition-app.appspot.com/')[1];
      await bucket.file(filePath).download({ destination: tempFilePath });

      // Process the image
      const imageBuffer = await sharp(tempFilePath)
        .resize(224, 224)
        .toBuffer();

      // Convert image to tensor and get embedding
      const tensor = tf.node.decodeImage(imageBuffer);
      const expandedTensor = tensor.expandDims(0);
      const normalizedTensor = expandedTensor.div(255.0);

      // Get face embedding
      const predictions = await model.predict(normalizedTensor);
      const faceEmbedding = predictions.arraySync()[0];

      // Store in Firestore
      await admin.firestore().collection('knownFaces').add({
        name,
        embedding: faceEmbedding,
        createdAt: admin.firestore.FieldValue.serverTimestamp()
      });

      // Clean up
      tensor.dispose();
      expandedTensor.dispose();
      normalizedTensor.dispose();
      predictions.dispose();

      res.status(200).json({ success: true, message: 'Face added to database' });
    } catch (error) {
      console.error('Error adding face to database:', error);
      res.status(500).send(`Error: ${error.message}`);
    }
  });
}); 