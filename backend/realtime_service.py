from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect  # Add this line
import asyncio
import cv2
import numpy as np
import base64
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import time
import mediapipe as mp  # Add this import

# Add src to path
print(f"Current working directory: {os.getcwd()}")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.realtime_inference import RealtimeEmotionPredictor

app = FastAPI()

# Initialize emotion predictor
predictor = None

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global predictor
    connection_active = True
    last_process_time = time.time()
    min_process_interval = 1.0 / 15.0  # 15 FPS maximum
    
    if not predictor:
        await websocket.accept()
        await websocket.close(code=1011)
        return

    await websocket.accept()
    
    try:
        while connection_active:
            try:
                # Rate limiting
                current_time = time.time()
                if current_time - last_process_time < min_process_interval:
                    await asyncio.sleep(0.001)
                    continue
                
                last_process_time = current_time
                
                # Process frame
                data = await websocket.receive_text()
                encoded_data = data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Process frame and detect face
                face = predictor.detect_face(frame)
                
                if face is not None:
                    x, y, w, h = face
                    face_tensor = predictor.preprocess_face(frame, face)
                    emotion, confidence = predictor.predict_emotion(face_tensor)
                    
                    # Simplified drawing
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"{emotion}: {confidence:.1%}"
                    cv2.putText(frame, text, (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                              (0, 255, 0), 2)

                # Optimize encoding
                _, buffer = cv2.imencode('.jpg', frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                await websocket.send_json({
                    "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
                    "emotion": emotion if face is not None else None,
                    "confidence": float(confidence) if face is not None else None
                })
                
            except WebSocketDisconnect:
                connection_active = False
                break
            except Exception as e:
                continue
                
    finally:
        print("WebSocket connection ended")
        if connection_active:
            try:
                await websocket.close()
            except:
                pass

# Add health check endpoint to verify predictor status
@app.get("/health")
async def health_check():
    if predictor is None:
        return {"status": "error", "message": "Predictor not initialized"}
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        # For local development, use relative path
        if os.environ.get('ENVIRONMENT') == 'development':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model.pth')
        else:
            model_path = '../model/best_model.pth'
            
        print(f"\nAttempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            # List contents of model directory
            model_dir = os.path.dirname(model_path)
            if os.path.exists(model_dir):
                print(f"Contents of {model_dir}:")
                print(os.listdir(model_dir))
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        print("Model file exists, creating predictor...")
        predictor = RealtimeEmotionPredictor(
            model_path=model_path,
            num_classes=5
        )
        print("Predictor initialized successfully")
        
    except Exception as e:
        import traceback
        print(f"Error initializing predictor: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
