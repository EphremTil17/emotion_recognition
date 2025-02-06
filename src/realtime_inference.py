import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import mediapipe as mp
import time
from model import EmotionRecognitionModel

class RealtimeEmotionPredictor:
    def __init__(self, model_path, num_classes=5):
        # Initialize device and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # Load model
        try:
            self.model = EmotionRecognitionModel.load_from_checkpoint(
                model_path, num_classes=num_classes
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
    def detect_face(self, frame):
        """Detect face using MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return None
        
        # Get first face detection
        detection = results.detections[0]
        ih, iw, _ = frame.shape
        
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        
        # Add padding (20%)
        padding = 0.2
        x = max(0, int(x - padding * w))
        y = max(0, int(y - padding * h))
        w = min(iw - x, int(w * (1 + 2 * padding)))
        h = min(ih - y, int(h * (1 + 2 * padding)))
        
        return (x, y, w, h)
        
    def preprocess_face(self, frame, face):
        """Preprocess detected face"""
        if face is None:
            return None
            
        x, y, w, h = face
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size == 0:
            return None
            
        face_tensor = self.transform(face_img)
        return face_tensor.unsqueeze(0)
        
    def predict_emotion(self, face_tensor):
        """Predict emotion using the model"""
        if face_tensor is None:
            return None, 0.0
            
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            outputs = self.model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            return self.emotions[prediction.item()], confidence.item()
            
    def draw_annotations(self, frame, face, emotion, confidence):
        """Draw bounding box and emotion prediction"""
        if face is None:
            return frame
            
        x, y, w, h = face
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Prepare and draw text
        text = f"{emotion}: {confidence:.2%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Background rectangle
        cv2.rectangle(frame, 
                     (x, y - text_size[1] - 10),
                     (x + text_size[0], y),
                     (0, 255, 0),
                     -1)
        
        # Text
        cv2.putText(frame,
                    text,
                    (x, y - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness)
                    
    def setup_camera(self):
        """Initialize and setup camera"""
        print("\nInitializing webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam (index 0)")
        
        # Get and print camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Camera properties:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        
        return cap
                    
    def run_webcam(self):
        """Run real-time emotion detection"""
        print("\nInitializing webcam...")
        cap = self.setup_camera()
        
        # Setup display window
        cv2.namedWindow('Emotion Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Emotion Recognition', 960, 540)
        
        # FPS tracking
        frame_times = []
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Process frame
                face = self.detect_face(frame)
                if face is not None:
                    face_tensor = self.preprocess_face(frame, face)
                    emotion, confidence = self.predict_emotion(face_tensor)
                    self.draw_annotations(frame, face, emotion, confidence)
                
                # Calculate and show FPS
                frame_times.append(time.time() - frame_start)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                if frame_times:
                    current_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    cv2.putText(frame,
                               f"FPS: {current_fps:.1f}",
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1,
                               (0, 255, 0),
                               2)
                
                # Display frame
                cv2.imshow('Emotion Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1)
                if key != -1:
                    key = key & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        frame_times = []
                
                # Check if window was closed
                if cv2.getWindowProperty('Emotion Recognition', cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except Exception as e:
            print(f"Error: {str(e)}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')
    parser.add_argument('--model', default='../model/best_model.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=5,
                      help='Number of emotion classes')
    
    args = parser.parse_args()
    
    try:
        predictor = RealtimeEmotionPredictor(args.model, args.num_classes)
        predictor.run_webcam()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()