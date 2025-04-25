"""
MNIST Training Script
This script trains the MNIST classification model with data augmentation and progress tracking.
"""

import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import MNISTModel
import datetime
from tqdm import tqdm
import logging
import os

def train_model():
    """
    Train the MNIST classification model.
    Includes data augmentation, progress tracking, and model saving.
    """
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===== IMAGE AUGMENTATION PIPELINE =====
    # This pipeline applies transformations to each image before training
    transform = transforms.Compose([
        # 1. Random rotation: Rotate image between -7 and 7 degrees
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        
        # 2. Random scaling: Scale image between 95% and 105% of original size
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        
        # 3. Random color adjustments: Vary brightness, contrast, saturation, and hue
        transforms.ColorJitter(
            brightness=0.10,  # Adjust brightness by ±10%
            contrast=0.1,     # Adjust contrast by ±10%
            saturation=0.10,  # Adjust saturation by ±10%
            hue=0.1          # Adjust hue by ±10%
        ),
        
        # 4. Convert PIL Image to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # ===== LOAD AND PREPARE TRAINING DATA =====
    # Create dataset with augmentation pipeline
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform  # Apply augmentation pipeline
    )
    
    # Create data loader with batch size 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    
    # ===== SAVE AUGMENTED IMAGE SAMPLES =====
    # Get first batch of augmented images
    images, _ = next(iter(train_loader))
    
    # Save the batch as a grid of 8x8 images (64 total)
    # Each image will show different augmentations
    save_image(images, 'transformed_batch.jpg', nrow=8, padding=2)
    
    # Initialize model and move to device
    model = MNISTModel().to(device)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params:,}")
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    # Set up logging with absolute path
    log_path = os.path.join(os.getcwd(), 'train.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(message)s',
        filemode='w'  # Overwrite existing log file
    )
    
    # Training loop
    model.train()
    total = 0
    correct = 0
    
    # Progress bar for training
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total += target.size(0)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # Print and log final accuracy in a consistent format
    final_accuracy = 100. * correct / total
    accuracy_str = f'Final Training Accuracy: {final_accuracy:.2f}%'
    print(f'\n{accuracy_str}')
    logging.info(accuracy_str)
    
    # Save trained model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'mnist_model_{timestamp}.pth')
    
    return model

if __name__ == "__main__":
    train_model() 