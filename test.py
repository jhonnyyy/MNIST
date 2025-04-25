import os
import torch
from torchvision import datasets, transforms
from model import MNISTModel
import pytest
import numpy as np

def test_model_architecture():
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, which exceeds the limit of 25000"
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    
    return model

def test_model_accuracy():
    model = test_model_architecture()
    model.eval()
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def test_model_output_range():
    """Test that model outputs are valid log probabilities"""
    model = MNISTModel()
    model.eval()
    
    # Generate random input
    test_input = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        output = model(test_input)
        
    # Check that outputs are log probabilities
    assert torch.all(output <= 0), "All outputs should be ≤ 0 (log probabilities)"
    assert torch.all(torch.isclose(torch.exp(output).sum(dim=1), torch.ones(1))), "Probabilities should sum to 1"

def test_model_robustness():
    """Test model's robustness to small input perturbations"""
    model = MNISTModel()
    model.eval()
    
    # Generate random input
    test_input = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        original_output = model(test_input)
        
        # Add small noise
        noisy_input = test_input + 0.1 * torch.randn_like(test_input)
        noisy_output = model(noisy_input)
        
        # Check that outputs are similar
        output_diff = torch.abs(original_output - noisy_output)
        assert torch.mean(output_diff) < 1.0, "Model should be robust to small input perturbations"

def test_model_gradient_flow():
    """Test that gradients flow properly through the network"""
    model = MNISTModel()
    model.train()
    
    # Generate random input and target
    test_input = torch.randn(1, 1, 28, 28, requires_grad=True)
    target = torch.randint(0, 10, (1,))
    
    # Forward pass
    output = model(test_input)
    loss = torch.nn.functional.nll_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are not zero
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Gradient for {name} is zero"

if __name__ == "__main__":
    accuracy = test_model_accuracy()
    print(f"Model accuracy: {accuracy:.2f}%") 