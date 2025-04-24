"""
MNIST Classification Model
This module implements a convolutional neural network for MNIST digit classification.
The architecture is optimized for minimal parameters while maintaining good accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    """
    A CNN model for MNIST digit classification.
    Architecture:
    - Three convolutional layers with increasing channels (6, 12, 16)
    - Each conv layer followed by GELU activation and batch normalization
    - Max pooling after first two conv layers
    - Two fully connected layers with dropout
    - Total parameters: ~21.5k
    """
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First convolutional block: 1->6 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2),
            
            # Second convolutional block: 6->12 channels
            nn.Conv2d(6, 12, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2),
            
            # Third convolutional block: 12->16 channels
            nn.Conv2d(12, 16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(16)
        )
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 24),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(24, 10)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Log probabilities for each digit class
        """
        x = self.conv1(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) 