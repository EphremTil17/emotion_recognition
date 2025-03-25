# EngageAi: Emotion Recognition System

A real-time and video emotion recognition system that analyzes facial expressions to detect emotions using Deep Learning, FastAPI, and React to create analytics about engagement levels in educational and presentation contexts.

## Overview

This system processes videos and webcam feeds to recognize five emotions (Angry, Happy, Neutral, Sad, Surprise) using a deep learning model (CNNs) based on computer vision techniques. The application provides both real-time analysis through webcam feeds (WebSockets) and batch processing for uploaded videos, with comprehensive analytics and content insights.

Data stays only for current session, gets wiped on reload.

## Technical Brief

* **Deep Learning Pipeline**: ResNet50-based CNN architecture trained on facial emotion datasets with mixed-precision training, achieving 97.7% validation accuracy across five emotion classes (Angry, Happy, Neutral, Sad, Surprise).

* **Real-time Processing**: Optimized WebSocket streaming with MediaPipe face detection integration, enabling 15-30 FPS emotion analysis with low latency on self-hosted hardware.

* **Multi-environment Analytics**: Scoring algorithms that contextualize emotional responses based on four distinct environments (classroom, product demo, seminar, general), with time-series analysis providing rolling 30-second, 1-minute, and cumulative engagement metrics.

* **Content-Emotion Correlation**: System for mapping emotional responses to specific content segments through timestamp correlation between Deepgram API transcriptions and detected emotions.

* **Microservices Architecture**: Modular FastAPI backend with specialized services that can scale independently (processing, real-time, content analysis, scholarly search), with seamless integration between PyTorch inference and React/MUI frontend.

* **GPU Acceleration**: CUDA optimization throughout the pipeline, from model training with mixed-precision to containerized inference with NVIDIA runtime support, 

## System Architecture

The project consists of four main components:

### 1. Core ML Components (`src/`)

The `src` directory contains the machine learning backbone of the system:

- **Model Architecture**: Uses a ResNet50-based model to classify emotions from facial images
- **Training Pipeline**: Scripts for training the emotion recognition model with data augmentation and validation
- **Inference Engines**:
  - `inference.py`: Base engine for static image processing
  - `realtime_inference.py`: Real-time emotion detection from webcam feeds using MediaPipe
  - `video_inference.py`: Processes video files to detect emotions frame-by-frame
  - `test_webcam.py` & `test_cuda.py`: Utilities for testing camera and CUDA availability

The model identifies five emotional states: Angry, Happy, Neutral, Sad, and Surprise.

### 2. Backend Services (`backend/`)

The backend is built with FastAPI and provides multiple services:

- **Processing Service (`processing_service.py`)**: Handles video file uploads, processes them using the ML model, and returns emotion analysis results
- **Realtime Service (`realtime_service.py`)**: Manages WebSocket connections for real-time webcam analysis
- **Content Analysis Service (`content_analysis_service.py`)**: Transcribes and analyzes speech content from videos using the Deepgram API
- **Scholarly Search Service (`scholarly_search.py`)**: Analyzes video content to find relevant academic resources using Google's Generative AI
- **API Layer (`api/main.py`)**: Provides a unified API interface with proper security controls

Key features:
- REST API for video uploads and processing
- WebSocket interface for real-time analysis
- Health check endpoints for monitoring
- Comprehensive analytics including emotion detection, engagement scoring, and content analysis
- Environment-specific scoring for different contexts (classroom, product demo, seminar)

Make sure to have a .env file in the backend folder with your Deepgram and Google Search/Gemini API Keys.

### 3. Frontend Application (`frontend/`)

A modern React application that provides the user interface:

- **Video Upload**: Interface for uploading videos to be processed with drag-and-drop support
- **Results Visualization**: Interactive charts and graphs to visualize emotion detection results and engagement metrics
- **Webcam Testing**: Real-time emotion detection through browser webcam access with camera selection
- **Content Analysis View**: Shows transcribed content with emotional context and engagement scores
- **Scholarly Resources**: Displays academic resources related to video content

Built with:
- React and Vite for a modern development experience
- Material UI for responsive, accessible UI components
- Recharts and ApexCharts for data visualization
- Custom styled components for enhanced user experience
- WebSocket integration for real-time analysis

### 4. Deployment and Infrastructure

- **Docker Configuration**: Multi-container setup with NVIDIA GPU support (Experimental)
- **Service Management**: Scripts for starting (`run_services.sh`) and stopping (`kill_processes.sh`) all services (Recommended)

## System Requirements

### Development Environment
- Python 3.11+
- CUDA 12.4+ and compatible NVIDIA GPU (for accelerated processing)
- Node.js 18+
- FFmpeg for video processing

### Deployment
- Docker and Docker Compose
- NVIDIA Container Runtime (for GPU support)
- 8GB+ RAM and/or 4GB+ GDDR4 GPU Memory recommended
- NVIDIA GPU with CUDA support (for optimal performance)

## Setup Instructions

### Option 1: Docker Deployment - Experimental

```bash
# Clone repository
git clone https://github.com/EphremTil17/emotion_recognition
cd emotion_recognition

# Start all services with Docker Compose
docker-compose -f docker/docker-compose.yml up -d
```

### Option 2: Manual Setup - Recommended

#### Backend Setup

```bash
# Clone and navigate to the repository
git clone https://github.com/EphremTil17/emotion_recognition
cd emotion_recognition

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
pip install -r requirements.txt

# Install backend dependencies
pip install -r backend/requirements.txt

# Start all services using the provided script
./run_services.sh

# Monitor logs in the newly created log directory
tail -f log/processing_service.log
tail -f log/realtime_service.log
tail -f log/content_analysis.log
tail -f log/scholarly_search.log
tail -f log/npm_run_dev.log
```

#### Stop All Services

```bash
./kill_processes.sh
```

#### Configuration

For content analysis and scholarly search features, create a `.env` file in the project root with:

```
DEEPGRAM_API_KEY=your_deepgram_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Usage

### Web Interface
1. Open your browser and navigate to `https://"yourdomain.in.vite"`
2. Upload a video file using the upload component or drag-and-drop interface
3. Wait for processing to complete
4. View the comprehensive analysis results:
   - Emotion distribution pie chart
   - Engagement scores for different environments
   - Time-series analysis of engagement levels
   - Content analysis with emotion correlation
   - Relevant scholarly resources
5. Alternatively, use the "Test Emotion Analyzer" button for real-time webcam analysis

## Development

### Model Training

To train the emotion recognition model:

```bash
# Prepare your dataset
python src/split_dataset.py --src_dir path/to/images --output_dir data/split_images

# Edit configuration if needed
nano src/config.yml

# Run training
python src/train.py
```

### Testing

```bash
# Test CUDA availability
python src/test_cuda.py

# Test webcam
python src/test_webcam.py

# Test inference on a single image
python src/inference.py path/to/image.jpg --model model/best_model.pth
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
│   ├── split_dataset.py    # Dataset preparation
│   ├── prepare_video.py    # Video preprocessing
│   ├── test_cuda.py        # CUDA testing utility
│   ├── test_webcam.py      # Webcam testing utility
│   ├── convert_model.py    # Model conversion utility
│   ├── evaluate.py         # Model evaluation
│   ├── utils.py            # Utilities and helpers
│   └── config.yml          # Training configuration
├── backend/                # API services
│   ├── processing_service.py # Video processing API
│   ├── realtime_service.py   # WebSocket service
│   ├── content_analysis_service.py # Content transcription using Deepgram
│   ├── scholarly_search.py  # Search content keywords for academic resources
│   ├── api/                 # API layer
│   │   └── main.py          # API endpoints with security controls
│   ├── uploads/            # Original videos storage
│   ├── processed/          # Processed videos output
│   ├── audio/              # Extracted audio for analysis
│   ├── analytics/          # Generated analytics files
│   └── requirements.txt    # Python dependencies
├── frontend/               # React application
│   ├── src/                # React components
│   │   ├── components/     # UI components
│   │   ├── services/       # API integration
│   │   ├── assets/         # Static assets
│   │   ├── App.jsx         # Main application
│   │   └── main.jsx        # Entry point
│   ├── public/             # Static assets
│   ├── vite.config.js      # Vite configuration
│   └── package.json        # JS dependencies
├── docker/                 # Deployment configs
│   ├── docker-compose.yml  # Service orchestration
│   ├── backend.Dockerfile  # Backend container
│   └── frontend.Dockerfile # Frontend container
├── run_services.sh         # Script to start all services
├── kill_processes.sh       # Script to stop all services
├── requirements.txt        # Core Python dependencies
└── README.md               # Project documentation
```

## Analytics Features - Extra

### Emotion Recognition
- Real-time facial emotion detection (Angry, Happy, Neutral, Sad, Surprise)
- Frame-by-frame video analysis with visualization
- Confidence scoring for detected emotions

### Engagement Metrics
- Overall engagement scoring (1-5 scale)
- Environment-specific engagement analysis:
  - Classroom: Optimized for educational settings
  - Product Demo: Focused on product presentation effectiveness
  - Seminar: Tailored for conference and lecture contexts
  - General: All-purpose engagement measurement

### Content Analysis
- Speech transcription from video audio
- Paragraph-level emotion correlation
- Engagement scoring per content segment
- Content summarization

### Academic Resources
- Automatic generation of relevant academic search queries
- Links to scholarly resources related to video content
- Research recommendations from multiple academic sources

## License

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007