import torch
import os
from tqdm import tqdm
import numpy as np

from .dataset import EmotionDataset
from .model import EmotionRecognitionModel
from .utils import Logger, plot_confusion_matrix, compute_metrics

def evaluate(config):
    # Initialize logger
    logger = Logger(config['log_dir'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {device}')
    
    # Create test dataset and dataloader
    test_dataset = EmotionDataset(
        data_dir=config['test_data_dir'],
        mode='test'
    )
    test_loader = test_dataset.get_dataloader(
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Load model
    num_classes = test_dataset.get_num_classes()
    model = EmotionRecognitionModel.load_from_checkpoint(
        config['checkpoint_path'],
        num_classes=num_classes
    )
    model = model.to(device)
    model.eval()
    
    # Evaluation loop
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute and log metrics
    metrics = compute_metrics(all_labels, all_predictions)
    logger.log(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    
    for i, class_acc in enumerate(metrics['per_class_accuracy']):
        logger.log(f"Class {test_dataset.classes[i]} Accuracy: {class_acc*100:.2f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels,
        all_predictions,
        test_dataset.classes,
        os.path.join(config['log_dir'], 'confusion_matrix.png')
    )

if __name__ == '__main__':
    config = {
        'test_data_dir': '../data/images/test',
        'log_dir': '../logs',
        'checkpoint_path': '../checkpoints/best_model.pth',
        'batch_size': 32,
    }
    
    evaluate(config)