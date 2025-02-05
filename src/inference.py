import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from model import EmotionRecognitionModel

class EmotionPredictor:
    def __init__(self, model_path, num_classes=5):
        # Load the face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load the emotion recognition model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmotionRecognitionModel.load_from_checkpoint(
            model_path, num_classes=num_classes
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms (same as validation transforms from dataset.py)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Emotion labels (make sure these match your training classes order)
        self.emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def detect_faces(self, image):
        """Detect single face in image using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(200, 200)
        )
            # If no face is detected with primary parameters, try backup parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(50, 50)
            )
        
        # If multiple faces detected, just take the first one
        # If no faces detected, return empty array
        return faces[:1]  # Return at most one face

    def preprocess_face(self, image, face):
        """Extract and preprocess face for model input"""
        x, y, w, h = face
        face_img = image[y:y+h, x:x+w]
        # Convert BGR to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        face_img = Image.fromarray(face_img)
        # Apply transforms
        face_tensor = self.transform(face_img)
        return face_tensor.unsqueeze(0)  # Add batch dimension

    def predict_emotion(self, face_tensor):
        """Predict emotion and confidence for a face tensor"""
        with torch.no_grad():
            outputs = self.model(face_tensor.to(self.device))
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            return self.emotions[prediction.item()], confidence.item()

    def process_image(self, image_path, output_path=None):
        """Process an image and visualize results"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            print("No faces detected in the image!")
            return
        
        # Process each face (will be only one based on our simplification)
        results = []
        for (x, y, w, h) in faces:
            # Preprocess face
            face_tensor = self.preprocess_face(image, (x, y, w, h))
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(face_tensor)
            results.append((emotion, confidence, (x, y, w, h)))
            
            # Draw thicker rectangle around face (increased thickness from 2 to 4)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
            
            # Add larger text with emotion and confidence
            text = f"{emotion}: {confidence:.2%}"
            # Increased font scale (from 0.9 to 1.5) and thickness (from 2 to 3)
            cv2.putText(
                image, text,
                (x, y-20),  # Moved text up slightly to accommodate larger font
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,        # Larger font scale
                (0, 255, 0),
                3          # Thicker text
            )
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Results saved to: {output_path}")
        else:
            # Display image
            cv2.imshow('Emotion Recognition', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print results
        print("\nDetected emotions:")
        for emotion, confidence, _ in results:
            print(f"- {emotion} (Confidence: {confidence:.2%})")

def main():
    parser = argparse.ArgumentParser(description='Emotion Recognition Inference')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--model', default='../checkpoints/best_model.pth',
                      help='Path to the model checkpoint')
    parser.add_argument('--output', help='Path to save the output image (optional)')
    parser.add_argument('--num-classes', type=int, default=5,
                      help='Number of emotion classes')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EmotionPredictor(args.model, args.num_classes)
    
    # Process image
    try:
        predictor.process_image(args.image_path, args.output)
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()