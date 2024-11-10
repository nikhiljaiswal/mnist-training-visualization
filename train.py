import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import MNISTNet
from tqdm import tqdm
import requests
import random

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model, loss, and optimizer
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training variables
epochs = 4
train_losses = []
train_accuracies = []
test_accuracies = []

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Calculate current accuracy and loss
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100 * correct / total
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
        
        # Send data to Flask server
        if batch_idx % 4 == 0:
            requests.post('http://localhost:9000/update', json={
                'loss': current_loss,
                'train_acc': current_acc,
                'epoch': epoch + 1
            })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    test_acc = test()
    
    train_losses.append(epoch_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f'\nEpoch {epoch+1}/{epochs}:')
    print(f'Training Loss: {epoch_loss:.4f}')
    print(f'Training Accuracy: {train_acc:.2f}%')
    print(f'Test Accuracy: {test_acc:.2f}%')

print("Training completed!")

# Test on 10 random images
model.eval()
test_images = []
test_labels = []
pred_labels = []

# Get 10 random test samples
indices = random.sample(range(len(test_dataset)), 10)
for idx in indices:
    image, label = test_dataset[idx]
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    test_images.append(image.cpu().squeeze().numpy())
    test_labels.append(label)
    pred_labels.append(predicted.item())

# Send final results to server
requests.post('http://localhost:9000/final_results', json={
    'images': [img.tolist() for img in test_images],
    'true_labels': test_labels,
    'pred_labels': pred_labels
}) 