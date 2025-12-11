import os
import random
import cv2

def visualize_labels(
    images_dir,
    labels_dir,
    num_samples=10,
    save_dir="visualized_labels"
):
    os.makedirs(save_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    
    if len(image_files) == 0:
        print("No images found!")
        return

    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for img_name in samples:
        image_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue

        h, w, _ = img.shape

        # Draw labels if they exist
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls, x, y, bw, bh = parts
                    cls = int(cls)
                    x, y, bw, bh = float(x), float(y), float(bw), float(bh)

                    # Convert YOLO format → pixel coords
                    x1 = int((x - bw / 2) * w)
                    y1 = int((y - bh / 2) * h)
                    x2 = int((x + bw / 2) * w)
                    y2 = int((y + bh / 2) * h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img, f"class {cls}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
        else:
            print(f"[WARN] No label for {img_name}")

        # Show the image
        cv2.imshow("Label Visualization", img)
        cv2.waitKey(500)  # show for 0.5 sec

        # Save visualization
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img)

    cv2.destroyAllWindows()
    print(f"Saved visualizations to: {save_dir}")


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGES_DIR = os.path.join(ROOT, "data", "Images")
    LABELS_DIR = os.path.join(ROOT, "data", "labels")

    visualize_labels(IMAGES_DIR, LABELS_DIR, num_samples=10)
