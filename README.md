# Emotion Recognition System

A real-time and video emotion recognition system using Deep Learning, FastAPI, and React.

## System Requirements

### Windows (Development/Processing Server)
- Python 3.11
- CUDA 11.8
- NVIDIA GPU (Tested on RTX 3060 Ti)
- Node.js 18+

### Linux (Deployment Server)
- Docker
- Docker Compose
- Nginx

## Project Structure
```
emotion_recognition/
├── backend/
│   ├── uploads/           # Original videos
│   ├── processed/         # Processed videos
│   ├── processing_service.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── package.json
└── docker/
    └── docker-compose.yml
```

## Setup Instructions

### Windows (Processing Server)
```bash
# Clone repository
git clone https://github.com/yourusername/emotion_recognition.git
cd emotion_recognition

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r backend/requirements.txt

# Start processing service
python backend/processing_service.py
```

### Frontend Development
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run start
```

### Linux Deployment
```bash
# Clone repository
git clone https://github.com/yourusername/emotion_recognition.git
cd emotion_recognition

# Start services
docker-compose up -d
```

## API Endpoints

### Processing Service (port 8001)
- POST `/process` - Process video file
- GET `/health` - Check service status

### Frontend (port 8000)
- Web interface for video upload and processing

## Development

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
```

## License

[Your chosen license]

## Contributors

- [Your Name]