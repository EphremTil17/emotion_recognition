# EngageAi: Emotion Recognition System

A real-time and video emotion recognition system that analyzes facial expressions to detect emotions using Deep Learning, FastAPI, and React to create analytics about engagement levels in session.

## Overview

This system processes videos and webcam feeds to recognize five emotions (Angry, Happy, Neutral, Sad, Surprise) using a deep learning model(CNNs) based on computer vision techniques. The application provides both real-time analysis through webcam feeds (websockets) and batch processing for uploaded videos.

## System Architecture

The project consists of three main components:

### 1. Core ML Components (`src/`)

The `src` directory contains the machine learning backbone of the system:

- **Model Architecture**: Uses a deep learning model to classify emotions from facial images
- **Training Pipeline**: Scripts for training the emotion recognition model
- **Inference Engines**:
  - `inference.py`: Base engine for static image processing
  - `realtime_inference.py`: Real-time emotion detection from webcam feeds using MediaPipe
  - `video_inference.py`: Processes video files to detect emotions frame-by-frame

Key files include:
- `model.py`: Defines the EmotionRecognitionModel class
- `dataset.py`: Handles dataset preprocessing and loading
- `train.py`: Training loop implementation with validation
- `config.yml`: Configuration parameters for model training

The model can identify five emotional states: Angry, Happy, Neutral, Sad, and Surprise.

### 2. Backend Service (`backend/`)

The backend is built with FastAPI and provides API endpoints for video processing:

- **Processing Service**: Handles video file uploads, processes them using the ML model, and returns emotion analysis results
- **Realtime Service**: Manages WebSocket connections for real-time webcam analysis
- **Analytics**: Generates engagement metrics and analytics from detected emotions

Key features:
- REST API for video uploads and processing
- WebSocket interface for real-time analysis
- Health check endpoints for monitoring

### 3. Frontend Application (`frontend/`)

A modern React application that provides the user interface:

- **Video Upload**: Interface for uploading videos to be processed
- **Results Visualization**: Charts and graphs to visualize emotion detection results
- **Webcam Testing**: Real-time emotion detection through browser webcam access

Built with:
- React for UI components
- Material UI for styling
- Recharts and ApexCharts for data visualization
- Axios for API communication

## System Requirements

### Development Environment
- Python 3.11+
- CUDA 11.8+ and compatible NVIDIA GPU
- Node.js 18+

### Deployment
- Docker and Docker Compose
- NVIDIA Container Runtime (for GPU support)

## Setup Instructions

### Option 1: Docker Deployment (Experimental)

```bash
# Clone repository
git clone https://github.com/EphremTil17/emotion_recognition
cd emotion_recognition

# Start all services with Docker Compose
docker-compose up -d
```

### Option 2: Manual Setup

#### Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install -r backend/requirements.txt

# Start backend services either usng the bash script or manually
cd emotion_recognition
python3 -m venv env
source env/bin/activate #Activate env
./run_services.sh


#To manually start the python servers and monitor them
python backend/processing_service.py
python backend/realtime_service.py


# Monitor logs in newly created log directory
cd log
tail -f log/realtime_service.log
tail -f npm_run_dev.log
tail -f processing_service.log
```
#### Stop Services

```
./kill_processes.sh
```
#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## Usage

### Web Interface
1. Open your browser and navigate to `http://localhost:8000` or the web interface you configured in VITE
2. Upload a video file using the upload component
3. View the emotion analysis results in the visualization panel
4. Alternatively, use the webcam feature for real-time analysis

### API Endpoints

#### Processing Service (port 8001)
- `POST /process` - Upload and process a video file
- `GET /health` - Check service health status
- `GET /analytics` - Retrieve analytics from latest processed video

#### Realtime Service (WebSocket)
- `ws://localhost:8002/ws` - WebSocket endpoint for real-time webcam analysis

## Development

### Model Training

To train the emotion recognition model:

```bash
python src/train.py
```

### Running Tests

```bash
# Backend tests
python -m pytest backend/tests

# Frontend tests
cd frontend
npm test
```

### Building for Production

```bash
# Frontend build
cd frontend
npm run build

# Docker build
docker-compose -f docker/docker-compose.yml build
```

## Project Structure

```
emotion_recognition/
├── src/                    # Core ML components
│   ├── model.py            # Model architecture
│   ├── dataset.py          # Dataset handling
│   ├── train.py            # Training logic
│   ├── inference.py        # Image inference
│   ├── video_inference.py  # Video processing
│   ├── realtime_inference.py # Webcam processing
│   └── config.yml          # Configuration
├── backend/                # API services
│   ├── processing_service.py # Video processing API
│   ├── realtime_service.py   # WebSocket service
│   ├── uploads/            # Original videos storage
│   ├── processed/          # Processed videos output
│   └── requirements.txt    # Python dependencies
├── frontend/               # React application
│   ├── src/                # React components
│   ├── public/             # Static assets
│   └── package.json        # JS dependencies
└── docker/                 # Deployment configs
    ├── docker-compose.yml  # Service orchestration
    ├── backend.Dockerfile  # Backend container
    └── frontend.Dockerfile # Frontend container
```

## License

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007]