import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 128
BATCH_SIZE = 32

# Paths
RANK_MODEL_PATH = "models/rank_classifier.pth"
SUIT_MODEL_PATH = "models/suit_classifier.pth"
RANK_DATA_DIR = "cropped_cards/rank"
SUIT_DATA_DIR = "cropped_cards/suit"

# Output folder
OUTPUT_DIR = "CNNData"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classes
RANK_CLASSES = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUIT_CLASSES = ["C", "D", "H", "S"]

# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# -------------------------
# Dataset & Loader function
# -------------------------
def get_loader(data_dir):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset.classes

# -------------------------
# Load CNN Model
# -------------------------
def load_cnn_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

rank_model = load_cnn_model(RANK_MODEL_PATH, len(RANK_CLASSES))
suit_model = load_cnn_model(SUIT_MODEL_PATH, len(SUIT_CLASSES))

# -------------------------
# Evaluation function
# -------------------------
def evaluate_model(model, loader, class_names, name="Model"):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n=== {name} Evaluation ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

# -------------------------
# Load datasets
# -------------------------
rank_loader, _ = get_loader(RANK_DATA_DIR)
suit_loader, _ = get_loader(SUIT_DATA_DIR)

# -------------------------
# Evaluate
# -------------------------
evaluate_model(rank_model, rank_loader, RANK_CLASSES, name="Rank_CNN")
evaluate_model(suit_model, suit_loader, SUIT_CLASSES, name="Suit_CNN")
