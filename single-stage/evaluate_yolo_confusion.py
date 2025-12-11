import os
import glob
import argparse

import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import yaml
except ImportError:
    yaml = None
    print("[WARN] PyYAML not installed – class names will default to indices.")


def load_class_names(data_yaml_path, num_classes=None):
    """
    Load class names from data.yaml if possible.
    Fallback: class0, class1, ...
    """
    if yaml is not None and os.path.exists(data_yaml_path):
        with open(data_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        names = data.get("names", None)
        if isinstance(names, dict):
            # e.g. {0: '10C', 1: '10D', ...}
            # sort by key
            names = [names[k] for k in sorted(names.keys())]
        if isinstance(names, list):
            return names

    # fallback
    if num_classes is None:
        num_classes = 53  # safe default for your dataset
    return [f"class{i}" for i in range(num_classes)]


def read_gt_class(label_path):
    """
    Read YOLO-format label file and return the first class id.
    Assumes ONE card per image (which is basically true in your data).
    """
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    cls_id = int(parts[0])
                    return cls_id
                except ValueError:
                    return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO single-stage model with confusion matrix.")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(root_dir, "runs", "single_stage", "real2", "weights", "best.pt"),
        help="Path to YOLO weights (.pt).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=os.path.join(root_dir, "data", "Images"),
        help="Path to images directory.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=os.path.join(root_dir, "data", "labels"),
        help="Path to labels directory.",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default=os.path.join(root_dir, "configs", "data.yaml"),
        help="Path to data.yaml (for class names).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., 'cpu' or '0').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(root_dir, "YOLO_eval"),
        help="Directory to save confusion matrix image.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading YOLO model from:", args.weights)
    model = YOLO(args.weights)

    # Collect image paths
    exts = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.images_dir, ext)))

    if not image_paths:
        print("[ERROR] No images found in:", args.images_dir)
        return

    print(f"[INFO] Found {len(image_paths)} images.")

    # First pass: figure out max class id from labels
    max_cls_id = -1
    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(args.labels_dir, base + ".txt")
        gt_cls = read_gt_class(lbl_path)
        if gt_cls is not None and gt_cls > max_cls_id:
            max_cls_id = gt_cls

    if max_cls_id < 0:
        print("[ERROR] No valid labels found in:", args.labels_dir)
        return

    num_classes = max_cls_id + 1
    class_names = load_class_names(args.data_yaml, num_classes=num_classes)
    if len(class_names) < num_classes:
        # pad if needed
        class_names += [f"class{i}" for i in range(len(class_names), num_classes)]

    print(f"[INFO] Using {num_classes} classes.")
    print("[INFO] Class names:", class_names)

    y_true = []
    y_pred = []

    total_images_with_labels = 0
    images_with_detection = 0
    images_with_no_det = 0

    # Loop over images
    for i, img_path in enumerate(sorted(image_paths)):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(args.labels_dir, base + ".txt")
        gt_cls = read_gt_class(lbl_path)

        if gt_cls is None:
            continue  # skip unlabeled images

        total_images_with_labels += 1

        # Run model
        results = model(
            img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            # No detections for this image
            images_with_no_det += 1
            continue

        # Take highest-confidence detection as predicted class
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(np.argmax(confs))

        pred_cls = int(cls_ids[best_idx])

        y_true.append(gt_cls)
        y_pred.append(pred_cls)
        images_with_detection += 1

        if (i + 1) % 200 == 0:
            print(f"[INFO] Processed {i+1}/{len(image_paths)} images...")

    print("\n[STATS]")
    print(f"Total labeled images: {total_images_with_labels}")
    print(f"Images with at least one detection: {images_with_detection}")
    print(f"Images with NO detections (ignored in confusion matrix): {images_with_no_det}")

    if len(y_true) == 0:
        print("[ERROR] No images had both labels and detections. Nothing to evaluate.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report (only on images where we had a detection)
    print("\n=== YOLO Single-Stage Evaluation (per-image top-1) ===")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("Confusion Matrix (raw counts):")
    print(cm)

    # Save arrays for metrics computation
    np.savetxt("y_true.txt", y_true, fmt="%d")
    np.savetxt("y_pred.txt", y_pred, fmt="%d")

    print("[INFO] Saved y_true.txt and y_pred.txt")


    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,   # set True if you want all numbers (may get crowded with 53 classes)
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("YOLO Single-Stage Confusion Matrix (per-image top-1)")

    out_path = os.path.join(args.output_dir, "yolo_single_stage_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\n[INFO] Confusion matrix figure saved to: {out_path}")


if __name__ == "__main__":
    main()
