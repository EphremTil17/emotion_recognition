import os
import shutil
import random
from pathlib import Path
import argparse

def split_dataset(src_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train, validation and test sets.
    
    Args:
        src_dir (str): Source directory containing emotion folders
        output_dir (str): Output directory for split dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        
    # Get all emotion classes
    emotion_classes = [d for d in os.listdir(src_dir) 
                      if os.path.isdir(os.path.join(src_dir, d))]
    
    for emotion in emotion_classes:
        # Get all images for this emotion
        emotion_dir = os.path.join(src_dir, emotion)
        images = [f for f in os.listdir(emotion_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        # Calculate split sizes
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to respective directories
        splits_dict = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split, split_images in splits_dict.items():
            # Create emotion directory in split
            split_emotion_dir = os.path.join(output_dir, split, emotion)
            os.makedirs(split_emotion_dir, exist_ok=True)
            
            # Copy images
            for img in split_images:
                src_path = os.path.join(src_dir, emotion, img)
                dst_path = os.path.join(split_emotion_dir, img)
                shutil.copy2(src_path, dst_path)
        
        print(f"Processed {emotion}:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Validation: {len(val_images)} images")
        print(f"  Test: {len(test_images)} images")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train, validation and test sets')
    parser.add_argument('--src_dir', type=str, default='../data/images',
                      help='Source directory containing emotion folders')
    parser.add_argument('--output_dir', type=str, default='../data/split_images',
                      help='Output directory for split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                      help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                      help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                      help='Ratio of test data')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Verify ratios sum to 1
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    
    split_dataset(
        args.src_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

if __name__ == '__main__':
    main()