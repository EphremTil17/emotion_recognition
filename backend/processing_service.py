from pathlib import Path
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root / "src"))

import torch
from fastapi import FastAPI, File, UploadFile
import uvicorn
import tempfile
import os
from video_inference import VideoEmotionPredictor
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global predictor
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.set_device(0)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device('cpu')
            logger.info("CUDA not available. Using CPU.")
        
        # Initialize predictor
        predictor = VideoEmotionPredictor(
            model_path=model_path,
            num_classes=5,
            target_fps=30
        )
        logger.info(f"Predictor initialized on device: {predictor.device}")
        
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    yield
    
    # Cleanup
    if predictor and hasattr(predictor, 'model'):
        del predictor.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# Initialize the emotion predictor
model_path = "../model/best_model.pth"  # Adjust path as needed
predictor = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(predictor.device) if predictor else "not initialized",
        "memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
    }

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
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = file.filename
        filename_base = f"{timestamp}_{original_filename}"
        
        # Define paths for uploaded and processed videos
        upload_path = UPLOAD_DIR / filename_base
        processed_path = PROCESSED_DIR / f"processed_{filename_base}"
        
        # Save uploaded file
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        print(f"Original video saved to: {upload_path}")
        
        # Process the video
        emotion_counts, frames_processed = predictor.process_video(
            str(upload_path), 
            str(processed_path)
        )
        
        print(f"Processed video saved to: {processed_path}")
        
        return {
            "status": "success",
            "frames_processed": frames_processed,
            "emotion_counts": emotion_counts,
            "original_path": str(upload_path),
            "processed_path": str(processed_path)
        }
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("processing_service:app", 
                host="0.0.0.0", 
                port=8001, 
                reload=True)