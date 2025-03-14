from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import httpx
import os
from dotenv import load_dotenv
import aiofiles
import uuid
from pathlib import Path
import mimetypes

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Emotion Recognition API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROCESSING_SERVER = os.getenv("PROCESSING_SERVER", "http://localhost:8001")
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

@app.get("/")
async def read_root():
    return {"status": "online", "message": "Emotion Recognition API"}

@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)):
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        original_extension = Path(file.filename).suffix
        temp_path = UPLOAD_DIR / f"{file_id}{original_extension}"
        
        # Save uploaded file
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Send to processing server
        async with httpx.AsyncClient() as client:
            files = {'file': (file.filename, open(temp_path, 'rb'))}
            response = await client.post(
                f"{PROCESSING_SERVER}/process",
                files=files,
                timeout=600.0  # 10 minutes timeout
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Processing failed")
            
            result = response.json()
            
        return JSONResponse({
            "status": "success",
            "message": "Video processed successfully",
            "job_id": file_id,
            "download_url": f"/api/download/{file_id}{original_extension}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/download/{filename}")
async def download_video(filename: str):
    # Use basename to strip any path components from the filename and Verify the constructed path is within the allowed directory
    safe_filename = os.path.basename(filename)
    
    # Construct the file path using the sanitized filename
    file_path = PROCESSED_DIR / safe_filename
     
    try:
        # Resolve to get the canonical path with symlinks resolved
        real_path = file_path.resolve(strict=False)
        real_processed_dir = PROCESSED_DIR.resolve()
        
        # Check that the file's path starts with the processed directory path
        if not str(real_path).startswith(str(real_processed_dir)):
            raise HTTPException(status_code=400, detail="Invalid file path")
    except (FileNotFoundError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Now check if the file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine the appropriate media type instead of hardcoding
    media_type, _ = mimetypes.guess_type(str(file_path))
    if not media_type:
        media_type = "application/octet-stream"  # Default fallback
        
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=safe_filename  # Use the sanitized filename here too
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)