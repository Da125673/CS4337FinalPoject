import os

img_dir = "Images/Images"
label_dir = "Images/labels"

images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
labels = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(".txt")])

missing = []

for img in images:
    base = os.path.splitext(img)[0]
    expected_label = base + ".txt"
    if expected_label not in labels:
        missing.append(img)

print("Total images:", len(images))
print("Total labels:", len(labels))
print("Images missing a label:", len(missing))
print("Examples:", missing[:10])
