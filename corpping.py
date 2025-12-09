import os
import json
import cv2
from pathlib import Path

# -------------------------
# Config
# -------------------------
JSON_FILE = "annotation.json"         # Your JSON file
IMG_DIR = "Images/Images"                    # Directory with your images
YOLO_LABEL_DIR = "YOLO_Annotations/YOLO_Annotations"        # Directory with YOLO .txt files
OUTPUT_DIR = "cropped_cards"          # Where cropped cards will be saved

# Cropping output structure: by rank and by suit
RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
SUITS = ['C','D','H','S']             # Clubs, Diamonds, Hearts, Spades

Path(OUTPUT_DIR).mkdir(exist_ok=True)

for r in RANKS:
    Path(os.path.join(OUTPUT_DIR, "rank", r)).mkdir(parents=True, exist_ok=True)
for s in SUITS:
    Path(os.path.join(OUTPUT_DIR, "suit", s)).mkdir(parents=True, exist_ok=True)

# -------------------------
# Functions
# -------------------------
def yolo_to_bbox(x_c, y_c, w, h, img_w, img_h):
    """Convert YOLO normalized coordinates to pixel bounding box."""
    x_center = x_c * img_w
    y_center = y_c * img_h
    width = w * img_w
    height = h * img_h
    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)
    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)
    # Clip to image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w - 1, xmax)
    ymax = min(img_h - 1, ymax)
    return xmin, ymin, xmax, ymax

def extract_rank_suit(filename):
    """Extract rank and suit from filename like '10D17.jpg' or 'AS38.jpg'"""
    name = os.path.splitext(filename)[0]  # remove extension
    # Handle rank: can be 1 or 2 chars
    if name[0] == 'A' or name[0] == 'J' or name[0] == 'Q' or name[0] == 'K':
        rank = name[0]
        suit = name[1]
    elif name[0:2] == '10':
        rank = '10'
        suit = name[2]
    else:
        rank = name[0]
        suit = name[1]
    return rank, suit

# -------------------------
# Main Processing
# -------------------------
with open(JSON_FILE, "r") as f:
    data = json.load(f)

for img_info in data['images']:
    filename = img_info['file_name']
    img_path = os.path.join(IMG_DIR, filename)
    
    if not os.path.exists(img_path):
        print(f"[WARNING] Image not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    # YOLO label file
    txt_file = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(YOLO_LABEL_DIR, txt_file)
    
    if not os.path.exists(txt_path):
        print(f"[WARNING] YOLO label not found: {txt_path}")
        continue

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.strip() == "":
            continue
        parts = line.strip().split()
        cls_id = parts[0]  # class id (can ignore for now)
        x_c, y_c, w, h = map(float, parts[1:5])
        xmin, ymin, xmax, ymax = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
        
        cropped = img[ymin:ymax, xmin:xmax]
        if cropped.size == 0:
            continue

        # Extract rank and suit
        rank, suit = extract_rank_suit(filename)

        # Save by rank
        rank_path = os.path.join(OUTPUT_DIR, "rank", rank, f"{os.path.splitext(filename)[0]}_{idx}.jpg")
        cv2.imwrite(rank_path, cropped)

        # Save by suit
        suit_path = os.path.join(OUTPUT_DIR, "suit", suit, f"{os.path.splitext(filename)[0]}_{idx}.jpg")
        cv2.imwrite(suit_path, cropped)

    print(f"[OK] Processed {filename} with {len(lines)} boxes")

print("All done! Cropped card dataset saved in:", OUTPUT_DIR)
