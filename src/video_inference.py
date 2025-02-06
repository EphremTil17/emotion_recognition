import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
import argparse
from pathlib import Path
import time
import mediapipe as mp

from model import EmotionRecognitionModel

class VideoEmotionPredictor:
    def __init__(self, model_path, num_classes=5, target_fps=4):
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for long-range
            min_detection_confidence=0.5
        )
        
        # Load the emotion recognition model
        self.model = EmotionRecognitionModel.load_from_checkpoint(
            model_path, num_classes=num_classes
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.target_fps = target_fps
        
    def detect_face(self, frame):
        """Detect face using MediaPipe"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return []
        
        # Get the first face detection
        detection = results.detections[0]
        ih, iw, _ = frame.shape
        
        # Get bounding box coordinates
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        
        # Add padding to the bounding box
        padding = 0.2  # 20% padding
        x = max(0, int(x - padding * w))
        y = max(0, int(y - padding * h))
        w = min(iw - x, int(w * (1 + 2 * padding)))
        h = min(ih - y, int(h * (1 + 2 * padding)))
        
        return [(x, y, w, h)]
        
    def preprocess_face(self, frame, face):
        """Extract and preprocess face for model input"""
        x, y, w, h = face
        # Ensure coordinates are within frame boundaries
        x, y = max(0, x), max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)
        
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:  # Check if the face image is empty
            return None
            
        face_tensor = self.transform(face_img)
        return face_tensor.unsqueeze(0)
        
    def predict_emotion(self, face_tensor):
        """Predict emotion and confidence"""
        with torch.no_grad():
            outputs = self.model(face_tensor.to(self.device))
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            return self.emotions[prediction.item()], confidence.item()
            
    def draw_annotations(self, frame, face, emotion, confidence):
        """Draw bounding box and emotion prediction on frame"""
        x, y, w, h = face
        
        # Draw thick rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        
        # Prepare text
        text = f"{emotion}: {confidence:.2%}"
        
        # Calculate text size and position
        font_scale = 1.5
        thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(frame, 
                     (x, y - text_size[1] - 20),
                     (x + text_size[0], y),
                     (0, 255, 0),
                     -1)  # Filled rectangle
        
        # Draw text
        cv2.putText(frame,
                    text,
                    (x, y - 10),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    thickness)
                    
    def process_video(self, input_path, output_path):
        """Process video file and create annotated output"""
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate process_every_n_frames
        process_every_n_frames = max(1, int(original_fps / self.target_fps))
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        # Get dimensions and check orientation
        height, width = frame.shape[:2]
        if height > width:
            print("\nWarning: For best results, please record video in landscape orientation!")
            print("Current dimensions:", width, "x", height)
            print("Tip: Hold your phone sideways when recording.\n")
        
        # Reset video capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Initialize video writer with original dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.target_fps,
            (width, height)
        )
        
        # Initialize emotion statistics
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        frames_processed = 0
        faces_detected = 0  # Add counter for face detections
        
        # Process frames
        pbar = tqdm(total=total_frames, desc="Processing video")
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            pbar.update(1)
            
            # Process only every nth frame
            if frame_count % process_every_n_frames != 0:
                continue
            
            # Detect and process face
            faces = self.detect_face(frame)
            if faces:
                faces_detected += 1  # Increment face detection counter
                face = faces[0]
                face_tensor = self.preprocess_face(frame, face)
                if face_tensor is not None:
                    emotion, confidence = self.predict_emotion(face_tensor)
                    
                    # Update statistics
                    emotion_counts[emotion] += 1
                    frames_processed += 1
                    
                    # Draw annotations
                    self.draw_annotations(frame, face, emotion, confidence)
                    
                    # Add statistics to frame
                    stats_y = 30
                    for emotion, count in emotion_counts.items():
                        if frames_processed > 0:
                            percentage = (count / frames_processed) * 100
                            stats_text = f"{emotion}: {percentage:.1f}%"
                            cv2.putText(frame, stats_text, (10, stats_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            stats_y += 25
            
            # Write frame
            out.write(frame)
            
        # Clean up
        pbar.close()
        cap.release()
        out.release()
        
        # Print final statistics
        print("\nProcessing Summary:")
        print(f"Total frames processed: {frame_count}")
        print(f"Faces detected in {faces_detected} frames")
        print(f"Face detection rate: {(faces_detected/frame_count)*100:.1f}%")
        
        print("\nEmotion Distribution:")
        for emotion, count in emotion_counts.items():
            if frames_processed > 0:
                percentage = (count / frames_processed) * 100
                print(f"{emotion}: {percentage:.1f}%")
                
        return emotion_counts, frames_processed
    
def main():
    parser = argparse.ArgumentParser(description='Video Emotion Recognition')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output', default='output.mp4',
                      help='Path to output video file')
    parser.add_argument('--model', default='../model/best_model.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--fps', type=int, default=4,
                      help='Target processing frame rate')
    parser.add_argument('--num-classes', type=int, default=5,
                      help='Number of emotion classes')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = VideoEmotionPredictor(
        args.model,
        num_classes=args.num_classes,
        target_fps=args.fps
    )
    
    # Process video
    try:
        start_time = time.time()
        emotion_counts, frames_processed = predictor.process_video(
            args.input_video,
            args.output
        )
        processing_time = time.time() - start_time
        
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Processed {frames_processed} frames")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()