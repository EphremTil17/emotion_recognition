import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights 

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EmotionRecognitionModel, self).__init__()
        
        # Load pre-trained ResNet-50
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path, num_classes):
        model = EmotionRecognitionModel(num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model