import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
ROOT = "images"   # contains images/ and labels/
IMG_DIR = os.path.join(ROOT, "images")
LBL_DIR = os.path.join(ROOT, "labels")

# YOLO dataset folder
DATASET = "dataset"
IMG_TRAIN = os.path.join(DATASET, "images/train")
IMG_VAL   = os.path.join(DATASET, "images/val")
LBL_TRAIN = os.path.join(DATASET, "labels/train")
LBL_VAL   = os.path.join(DATASET, "labels/val")

# Create folders
for path in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(path, exist_ok=True)

# Get all images
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Split train/val
train_imgs, val_imgs = train_test_split(image_files, test_size=0.2, random_state=42)

def move_files(img_list, img_dest, lbl_dest):
    for img_name in img_list:
        label_name = os.path.splitext(img_name)[0] + ".txt"

        img_src = os.path.join(IMG_DIR, img_name)
        lbl_src = os.path.join(LBL_DIR, label_name)

        img_out = os.path.join(img_dest, img_name)
        lbl_out = os.path.join(lbl_dest, label_name)

        if os.path.exists(img_src):
            shutil.copy(img_src, img_out)
        else:
            print(f"[WARNING] Missing image: {img_src}")

        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_out)
        else:
            print(f"[WARNING] Missing label: {lbl_src}")

# Move files
move_files(train_imgs, IMG_TRAIN, LBL_TRAIN)
move_files(val_imgs, IMG_VAL, LBL_VAL)

print("\n=== YOLO DATASET READY! ===")
print(f"Images train: {len(os.listdir(IMG_TRAIN))}")
print(f"Images val:   {len(os.listdir(IMG_VAL))}")
print(f"Labels train: {len(os.listdir(LBL_TRAIN))}")
print(f"Labels val:   {len(os.listdir(LBL_VAL))}")
