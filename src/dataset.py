import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class EmotionDataset:
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        
        # Define normalization parameters (ImageNet stats)
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        
        # Define transforms with additional augmentations for training
        self.transform = self._get_transforms()
        
        # Create dataset
        try:
            self.dataset = datasets.ImageFolder(
                root=self.data_dir,
                transform=self.transform
            )
            self.class_to_idx = self.dataset.class_to_idx
            self.classes = self.dataset.classes
        except Exception as e:
            print(f"Error loading dataset from {self.data_dir}: {str(e)}")
            raise
        
    def _get_transforms(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),  # Resize larger for random crops
                transforms.RandomCrop(224),     # Random crops for better generalization
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
    
    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0, **kwargs):
        """
        Create a DataLoader for the dataset.
        
        Args:
            batch_size (int): How many samples per batch to load
            shuffle (bool): Whether to shuffle the data
            num_workers (int): How many subprocesses to use for data loading
            **kwargs: Additional arguments to pass to DataLoader
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )

    def get_num_classes(self):
        return len(self.classes)