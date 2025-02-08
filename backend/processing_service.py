from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
import os
from datetime import datetime
import torch
from contextlib import asynccontextmanager

# Add the src directory to Python path
current_dir = Path(__file__).parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from src.video_inference import VideoEmotionPredictor

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the emotion predictor
model_path = "../model/best_model.pth"
predictor = None

@app.get("/health")
async def health_check():
    """Health check endpoint that's available immediately"""
    return {"status": "healthy"}
    
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

# Define paths relative to the backend directory
BACKEND_DIR = Path(__file__).parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = BACKEND_DIR / "processed"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    if not predictor:
        return {"error": "Predictor not initialized"}
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{timestamp}_{file.filename}"
        upload_path = UPLOAD_DIR / filename_base
        processed_path = PROCESSED_DIR / f"processed_{filename_base}"
        
        with open(upload_path, "wb") as f:
            f.write(await file.read())
        
        print(f"\nProcessing video: {upload_path}")
        emotion_counts, frames_processed = predictor.process_video(
            str(upload_path), 
            str(processed_path)
        )
        
        return {
            "status": "success",
            "frames_processed": frames_processed,
            "emotion_counts": emotion_counts,
            "processed_path": str(processed_path)
        }
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)