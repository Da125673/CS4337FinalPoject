import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_MODEL_PATH = "yolov8n.pt"
RANK_MODEL_PATH = "models/rank_classifier.pth"
SUIT_MODEL_PATH = "models/suit_classifier.pth"

RANK_DIR = "cropped_cards/rank"
SUIT_DIR = "cropped_cards/suit"

INPUT_SIZE = 128

# -------------------------
# Load actual class names
# -------------------------c
RANK_CLASSES = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUIT_CLASSES = ["C", "D", "H", "S"]

print("Rank classes:", RANK_CLASSES)
print("Suit classes:", SUIT_CLASSES)

# -------------------------
# Load YOLO Detector
# -------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

# -------------------------
# Load CNN Models
# -------------------------
def load_cnn_model(model_path, num_classes):
    model = models.resnet18(weights=None)  # IMPORTANT: use no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

rank_model = load_cnn_model(RANK_MODEL_PATH, len(RANK_CLASSES))
suit_model = load_cnn_model(SUIT_MODEL_PATH, len(SUIT_CLASSES))

# -------------------------
# Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# -------------------------
# Prediction Function
# -------------------------
def predict_card(crop):
    # Convert OpenCV → PIL
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        r_out = rank_model(tensor)
        s_out = suit_model(tensor)

    rank_idx = r_out.argmax(1).item()
    suit_idx = s_out.argmax(1).item()

    return RANK_CLASSES[rank_idx], SUIT_CLASSES[suit_idx]


# -------------------------
# Start Webcam
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not available.")
    exit()

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # -------------------------
    # YOLO Detection
    # -------------------------
    results = yolo_model(frame)

    if len(results[0].boxes) == 0:
        cv2.imshow("Card Detection", frame)
        if cv2.waitKey(1) & 0xFF == 'q':
            break
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()

    # -------------------------
    # Process each detected card
    # -------------------------
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1
    
        # Filter out non-card boxes
        if width > frame.shape[1] * 0.6 or height > frame.shape[0] * 0.6:
            continue
        if width < 50 or height < 50:
            continue
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 0.8:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        rank, suit = predict_card(crop)
        label = f"{rank} of {suit}"

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)


    cv2.imshow("Card Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
