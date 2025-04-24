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

def train_model():
    """
    Train the MNIST classification model.
    Includes data augmentation, progress tracking, and model saving.
    """
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data augmentation pipeline
    transform = transforms.Compose([
        # Random rotation between -7 and 7 degrees
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        # Random scaling between 95% and 105%
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        # Random color adjustments
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.1,
            saturation=0.10,
            hue=0.1
        ),
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load and prepare training data
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    
    # Save sample of augmented images
    images, _ = next(iter(train_loader))
    save_image(images, 'transformed_batch.jpg')
    
    # Initialize model and move to device
    model = MNISTModel().to(device)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params:,}")
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    # Set up logging
    logging.basicConfig(
        filename='train.log',
        level=logging.INFO,
        format='%(message)s'
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
    
    # Print final accuracy
    final_accuracy = 100. * correct / total
    print(f'\nFinal Training Accuracy: {final_accuracy:.2f}%')
    
    # Log final accuracy
    logging.info(f'Final Training Accuracy: {final_accuracy:.2f}%')
    
    # Save trained model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'mnist_model_{timestamp}.pth')
    
    return model

if __name__ == "__main__":
    train_model() 