import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
import random

# Path to your dataset
DATA_DIR = "140k real and fake faces/real_vs_fake/real-vs-fake"

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load full datasets
full_train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
full_valid_data = datasets.ImageFolder(f"{DATA_DIR}/valid", transform=transform)

# Take a small random slice for a quick test run
random.seed(42)
train_indices = random.sample(range(len(full_train_data)), 5000)
valid_indices = random.sample(range(len(full_valid_data)), 1000)

train_data = Subset(full_train_data, train_indices)
valid_data = Subset(full_valid_data, valid_indices)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)

print(f"Training images: {len(train_data)}")
print(f"Validation images: {len(valid_data)}")

# Load pretrained ResNet18 and adjust final layer for 2 classes
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Model ready. Starting training...")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} done. Accuracy: {accuracy:.2f}%")
        # Check validation accuracy
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "deepfake_detector.pth")
print("Model saved as deepfake_detector.pth")