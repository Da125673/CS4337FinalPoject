import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------------
# Config
# -------------------------

# Choose dataset: "rank" or "suit"
DATASET_TYPE = "suit"   # set to "suit" for suit classification
DATA_DIR = os.path.join("cropped_cards", DATASET_TYPE)

# -------------------------
# Data Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


# -------------------------
# Datasets & Dataloaders
# -------------------------
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
num_classes = len(dataset.classes)
print(f"Classes ({num_classes}): {dataset.classes}")

# Split into 80% train, 20% val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Model
# -------------------------
# Using pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # adjust final layer
model = model.to(DEVICE)

# -------------------------
# Loss & Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Acc: {val_acc:.4f}")

# -------------------------
# Save the model
# -------------------------
torch.save(model.state_dict(), f"{DATASET_TYPE}_classifier.pth")
print("Model saved!")
