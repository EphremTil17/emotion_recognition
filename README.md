# Facial Emotion Recognition with ResNet-50

This project implements a facial emotion recognition system using a pre-trained ResNet-50 model with PyTorch.

## Project Structure

```
emotion_recognition/
├── data/
│   └── images/           # Contains subfolders for each emotion
├── src/
│   ├── __init__.py
│   ├── dataset.py       # Dataset and data loading utilities
│   ├── model.py         # Model architecture definition
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   └── utils.py         # Helper functions
├── notebooks/
├── logs/                # Training logs and visualizations
├── checkpoints/         # Model checkpoints
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Place your emotion images in subfolders under `data/images/`
   - Each subfolder should be named after the emotion class (e.g., 'happy', 'sad', etc.)
   - Split your data into train/val/test sets

## Training

To train the model:

```bash
cd src
python train.py
```

You can modify the training configuration in the `train.py` script.

## Evaluation

To evaluate a trained model:

```bash
cd src
python evaluate.py
```

## Model Architecture

- Base model: ResNet-50 (pre-trained on ImageNet)
- Modified final layer for emotion classification
- Data augmentation:
  - Random horizontal flip
  - Color jitter
  - Random rotation
  - Normalization using ImageNet statistics

## Results

After training, you can find:
- Training logs in the `logs/` directory
- Model checkpoints in the `checkpoints/` directory
- Confusion matrix and evaluation metrics in the logs

## License

[Your chosen license]

# Video Requirements

For best results when using video emotion recognition:

1. Record videos in landscape orientation (horizontal)
2. Ensure good lighting conditions
3. Face should be clearly visible
4. Recommended resolution: 720p or higher
5. If you have a portrait video, use the prepare_video.py script:
   ```bash
   python prepare_video.py your_video.mp4
   ```

This will create a properly oriented version of your video ending in '_landscape.mp4'