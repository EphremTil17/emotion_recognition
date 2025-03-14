from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import re
import io
import logging
from contextlib import redirect_stdout

# Add the src directory to Python path
current_dir = Path(__file__).parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from src.video_inference import VideoEmotionPredictor

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://engageai.ephremst.com",
        "https://api.ephremst.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize the emotion predictor
model_path = "../model/best_model.pth"
predictor = None

# Define paths relative to the backend directory
BACKEND_DIR = Path(__file__).parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = BACKEND_DIR / "processed"
ANALYTICS_DIR = BACKEND_DIR / "analytics"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
ANALYTICS_DIR.mkdir(exist_ok=True)

def parse_processing_output(output_text):
    """Parse the console output to extract processing statistics"""
    stats = {}
    
    # Extract frames processed
    frames_match = re.search(r'Total frames processed: (\d+)', output_text)
    if frames_match:
        stats['frames_processed'] = int(frames_match.group(1))
    
    # Extract face detection rate
    detection_rate_match = re.search(r'Face detection rate: ([\d.]+)%', output_text)
    if detection_rate_match:
        stats['face_detection_rate'] = float(detection_rate_match.group(1))
    
    # Extract emotion distribution
    emotion_section = output_text.split('Emotion Distribution:')[-1].strip()
    emotion_lines = emotion_section.split('\n')
    emotion_distribution = {}
    
    for line in emotion_lines:
        if ':' in line:
            emotion, percentage = line.split(':')
            emotion = emotion.strip()
            percentage = float(percentage.strip().rstrip('%'))
            emotion_distribution[emotion] = percentage
    
    # Ensure all emotions are present
    all_emotions = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Undefined']
    for emotion in all_emotions:
        if emotion not in emotion_distribution:
            emotion_distribution[emotion] = 0.0
    
    stats['emotion_distribution'] = emotion_distribution
    
    return stats

def calculate_base_engagement(emotion):
    """Calculate base engagement value for an emotion (1-5 scale)"""
    base_scores = {
        "Happy": 4.5,   
        "Surprise": 5.0,
        "Neutral": 3.0, 
        "Sad": 2.0,    
        "Angry": 1.5     
    }
    return base_scores.get(emotion, 1.0)  # Default to 1.0 if emotion not found

def calculate_environment_score(base_score, emotion, environment="classroom"):
    """Apply environment-specific weights to base engagement score"""
    weights_mapping = {
        "classroom": {
            "Happy": 1.1,    # Normal weight in classroom
            "Surprise": 1.5,  # Slightly lower (might be distracted)
            "Neutral": 1.0,   # Higher (paying attention)
            "Sad": 0.8,      # Lower but not terrible
            "Angry": 0.6     # Significant reduction
        },
        "product_demo": {
            "Happy": 1.3,    # Much more important
            "Surprise": 1.2,  # Very good
            "Neutral": 0.7,   # Not good for demo
            "Sad": 0.5,      # Very bad
            "Angry": 0.4     # Extremely bad
        },
        "seminar": {
            "Happy": 0.9,    # Less important
            "Surprise": 1.2,  # Shows interest
            "Neutral": 1.1,   # Good for seminars
            "Sad": 0.7,      # Concerning
            "Angry": 0.6     # Very concerning
        },
        "general": {
            "Happy": 1.1,    # Baseline
            "Surprise": 1.2,  # Baseline
            "Neutral": 1.0,   # Baseline
            "Sad": 0.7,      # Below baseline
            "Angry": 0.6     # Well below baseline
        }
    }
    
    # Get environment weights
    weights = weights_mapping.get(environment, weights_mapping["general"])
    
    # Apply environment weight to base score
    weighted_score = base_score * weights.get(emotion, 1.0)
    
    # Ensure final score is between 1.0 and 5.0
    return max(1.0, min(5.0, weighted_score))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/analytics")
async def get_analytics():
    """Get the latest session analytics"""
    try:
        analytics_path = ANALYTICS_DIR / "latest_session.csv"
        if analytics_path.exists():
            return FileResponse(analytics_path)
        return {"error": "No analytics data available"}
    except Exception as e:
        logging.error(f"Error in get_analytics: {str(e)}")
        return {"error": "An internal error has occurred!"}

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        print("Initializing predictor...")
        predictor = VideoEmotionPredictor(
            model_path=model_path,
            num_classes=5,
            target_fps=30
        )
        print(f"Predictor initialized on device: {predictor.device}")
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Validate file size 
    MAX_SIZE = 100 * 1024 * 1024  # 100MB
    file_size = 0
    contents = await file.read(MAX_SIZE + 1)
    file_size = len(contents)
    
    if file_size > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Reset file position
    await file.seek(0)
    if not predictor:
        return {"error": "Predictor not initialized"}
    
    try:
        # Use fixed filenames to always overwrite previous files
        file_extension = Path(file.filename).suffix
        upload_filename = f"latest_upload{file_extension}"
        processed_filename = f"latest_processed{file_extension}"
        
        upload_path = UPLOAD_DIR / upload_filename
        processed_path = PROCESSED_DIR / processed_filename
        analytics_path = ANALYTICS_DIR / "latest_session.csv"
        
        # Clear the directories before saving new files
        for old_file in UPLOAD_DIR.glob("*.*"):
            old_file.unlink()
        
        for old_file in PROCESSED_DIR.glob("*.*"):
            old_file.unlink()
        
        # Save uploaded file
        with open(upload_path, "wb") as f:
            f.write(await file.read())
        
        print(f"\nProcessing video: {upload_path}")

        # Capture stdout to get processing statistics
        output = io.StringIO()
        with redirect_stdout(output):
            emotion_counts, frames_processed = predictor.process_video(
                str(upload_path), 
                str(processed_path)
            )
        
        output_text = output.getvalue()
        print(output_text)  # Print to console for logging
        
        # Parse the output
        stats = parse_processing_output(output_text)

        # Extract processing speed
        speed_match = re.search(r'(\d+.\d+)it/s', output_text)
        processing_speed = float(speed_match.group(1)) if speed_match else 0
        
        # Create time-series data that matches the distribution
        total_seconds = int(frames_processed / predictor.target_fps)
        time_range = np.arange(0, total_seconds, 0.5)  # Half-second intervals
        data = []

        # Parse emotion distribution for sampling
        emotions = list(stats['emotion_distribution'].keys())
        probabilities = [stats['emotion_distribution'][e]/100 for e in emotions]
        
        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p/total_prob for p in probabilities]
        else:
            # If all probabilities are 0, use uniform distribution
            probabilities = [1.0/len(emotions)] * len(emotions)
        
        # Generate timestamped data with environment-specific scores
        for t in time_range:
            # Generate emotion based on actual distribution
            emotion = np.random.choice(emotions, p=probabilities)
            confidence = np.random.uniform(0.8, 1.0)
            
            # Calculate base engagement score
            base_score = calculate_base_engagement(emotion)

            # Apply confidence scaling if needed
            if confidence < 0.8:
                base_score = max(1.0, base_score * confidence)

            # Calculate scores for each environment
            environment_scores = {
                env: calculate_environment_score(base_score, emotion, env)
                for env in ["classroom", "product_demo", "seminar", "general"]
            }

            data.append({
                'timestamp': round(t, 1),
                'emotion': emotion,
                'confidence': round(confidence, 2),
                'base_score': round(base_score, 1),  
                'classroom_score': round(environment_scores['classroom'], 4), 
                'product_demo_score': round(environment_scores['product_demo'], 4),  
                'seminar_score': round(environment_scores['seminar'], 4),  
                'general_score': round(environment_scores['general'], 4)  
            })

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Apply Moving Averages for each environment
        window_30s = 60  # 30 seconds at 2 measurements per second
        window_1min = 120  # 60 seconds at 2 measurements per second

        for env in ["classroom", "product_demo", "seminar", "general"]:
            score_column = f"{env}_score"
            df[f"{env}_30s_avg"] = df[score_column].rolling(window=window_30s, min_periods=1).mean().round(4)
            df[f"{env}_1min_avg"] = df[score_column].rolling(window=window_1min, min_periods=1).mean().round(4)
            df[f"{env}_cumulative"] = df[score_column].expanding().mean().round(4)

        # Save to CSV
        df.to_csv(analytics_path, index=False)
        
        # Compute overall engagement scores for all environments using averages
        environments = ["classroom", "product_demo", "seminar", "general"]
        engagement_scores = {}
        
        for env in environments:
            score_column = f"{env}_cumulative"
            if score_column in df.columns:
                # Use the final cumulative average for each environment
                engagement_scores[env] = float(df[score_column].iloc[-1])
            else:
                engagement_scores[env] = 1.0  # Default score if no data
        
        return {
            "status": "success",
            "frames_processed": stats['frames_processed'],
            "face_detection_rate": stats['face_detection_rate'],
            "emotion_distribution": stats['emotion_distribution'],
            "duration_seconds": len(time_range),
            "analytics_file": str(analytics_path),
            "engagement_scores": engagement_scores,
            "processed_video": str(processed_path)  # Include the path to the processed video
        }
            
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return {"error": "An internal error has occurred!"}

@app.get("/content-analysis")
async def get_content_analysis():
    """Get the latest content analysis"""
    try:
        content_analysis_path = ANALYTICS_DIR / "content_analysis.json"
        if content_analysis_path.exists():
            with open(content_analysis_path, "r") as f:
                return json.load(f)
        return {"error": "No content analysis available"}
    except Exception as e:
        logging.error(f"Error getting content analysis: {str(e)}")
        return {"error": "An internal error has occurred!"}
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)