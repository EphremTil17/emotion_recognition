FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p model src backend/processed backend/uploads

# Copy requirements first
COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy source code
COPY ./src /app/src
COPY ./backend /app/backend

WORKDIR /app/backend

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

CMD ["python3", "processing_service.py"]