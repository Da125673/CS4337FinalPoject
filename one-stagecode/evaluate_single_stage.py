# src/evaluate_single_stage.py

import os
from ultralytics import YOLO

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_config = os.path.join(root_dir, "configs", "data.yaml")

    weights = os.path.join(
        root_dir,
        "runs",
        "single_stage",
        "cards_single_stage",
        "weights",
        "best.pt",
    )

    model = YOLO(weights)

    metrics = model.val(
        data=data_config,
        imgsz=640,
        batch=16,
        split="val",   # or "test" if you configured that
    )

    print("mAP@0.50:", metrics.box.map50)
    print("mAP@0.50:0.95:", metrics.box.map)

if __name__ == "__main__":
    main()
