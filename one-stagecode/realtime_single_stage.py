# src/realtime_single_stage.py

import argparse
import os
import time

import cv2
from ultralytics import YOLO

CLASS_NAMES = [
    "10C", "10D", "10H", "10S",
    "2C", "2D", "2H", "2S",
    "3C", "3D", "3H", "3S",
    "4C", "4D", "4H", "4S",
    "5C", "5D", "5H", "5S",
    "6C", "6D", "6H", "6S",
    "7C", "7D", "7H", "7S",
    "8C", "8D", "8H", "8S",
    "9C", "9D", "9H", "9S",
    "AC", "AD", "AH", "AS",
    "JC", "JD", "JH", "JS",
    "KC", "KD", "KH", "KS",
    "QC", "QD", "QH", "QS"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time single-stage YOLO playing card detector"
    )

    # Default to your light run weights; you can change this later
    default_weights = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs",
        "single_stage",
        "real2",
        "weights",
        "best.pt",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=default_weights,
        help="Path to trained YOLO weights (.pt file)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (0 is default laptop webcam)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",  # CPU on your laptop
        help="Device for inference: 'cpu', '0', '1', etc.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detections (0-1)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for YOLO",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Could not find weights file: {args.weights}")

    print(f"[INFO] Loading YOLO model from: {args.weights}")
    model = YOLO(args.weights)

    print(f"[INFO] Opening webcam index {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {args.camera}")

    prev_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame from webcam.")
                break

            # Run YOLO on this frame
            results = model(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )

            r = results[0]

            # Draw detections
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    # xyxy: [x1, y1, x2, y2]
                    # Original YOLO box (corner patch)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    # Get frame size
                    h, w, _ = frame.shape

                    # Convert to float for math
                    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))

                    # Compute center + original box size
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bw = (x2 - x1)
                    bh = (y2 - y1)

                    # 🧩 Scale factor: how much bigger you want the “card” box to be
                    # Try 2.0 first; tweak to 1.8, 2.5, etc. after you see it.
                    scale_w = 2.0
                    scale_h = 2.5  # slightly taller than wide since cards are tall

                    new_w = bw * scale_w
                    new_h = bh * scale_h

                    # New expanded coords, clamped to frame bounds
                    x1 = max(0, int(cx - new_w / 2.0))
                    x2 = min(w - 1, int(cx + new_w / 2.0))
                    y1 = max(0, int(cy - new_h / 2.0))
                    y2 = min(h - 1, int(cy + new_h / 2.0))


                    if 0 <= cls_id < len(CLASS_NAMES):
                        card_name = CLASS_NAMES[cls_id]
                    else:
                        card_name = f"class{cls_id}"  # fallback, shouldn’t normally happen

                    label = card_name

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )

                    # Draw label background
                    (tw, th), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1) - th - baseline),
                        (int(x1) + tw, int(y1)),
                        (0, 255, 0),
                        thickness=-1,
                    )
                    # Label text
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), int(y1) - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Single-Stage Card Detector (YOLO)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
