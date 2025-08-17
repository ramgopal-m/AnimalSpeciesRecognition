from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
from torchvision import transforms
from model import AnimalSpeciesClassifier
import os

print("Starting server initialization...")

app = Flask(__name__)
CORS(app)

print("Loading model...")
# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = AnimalSpeciesClassifier.load_model('models/best_model.pth')
print("Model loaded successfully")
model = model.to(device)
model.eval()
print("Model ready for inference")

# Get class names in the same order as during training
classes = ['BEAR', 'CATS', 'CHEETAH', 'COW', 'CROCODILES', 'DEER', 'DOGS', 'ELEPHANT', 'GIRAFFE', 'GOAT', 'HIPPOPOTAMUS', 'HORSE', 'KANGAROO', 'LION', 'MEERKAT', 'MONKEY', 'MOOSE', 'OSTRICH', 'PANDA', 'PENGUINS', 'PORCUPINE', 'RABBIT', 'RHINOCEROS', 'SNAKE', 'SQUIRREL', 'TIGER', 'TORTOISE', 'WALRUS', 'WOLF', 'ZEBRA', 'antelope', 'buffalo', 'chimpanzee', 'collie', 'german+shepherd', 'grizzly+bear', 'otter', 'ox', 'persian+cat', 'seal']
print(f"Loaded classes: {classes}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    if 'image' not in request.files:
        print("No image file found in request")
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Read and preprocess the image
        print("Reading image file...")
        image = request.files['image'].read()
        print(f"Image size: {len(image)} bytes")
        
        print("Converting image to PIL format...")
        image = Image.open(io.BytesIO(image)).convert('RGB')
        print(f"Image dimensions: {image.size}")
        
        print("Applying transformations...")
        image = transform(image).unsqueeze(0).to(device)
        print(f"Transformed image shape: {image.shape}")

        # Make prediction
        print("Making prediction...")
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get the predicted class and confidence
            species = classes[predicted.item()]
            confidence = confidence.item()
            print(f"Predicted species: {species} with confidence: {confidence:.4f}")

        return jsonify({
            'species': species,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
