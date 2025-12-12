# src/train_single_stage.py

import os
from ultralytics import YOLO

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_config = os.path.join(root_dir, "configs", "data.yaml")

    # Tiny YOLO, good for low compute
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_config,
        epochs=10,                 # small takes forever
        imgsz=640,                # smaller image size => much faster
        batch=4,                  # smaller batch size for CPU / RAM
        name="real",
        project=os.path.join(root_dir, "runs", "single_stage"),
        workers=0,                
        patience=10,               # early stop if no improvement
        lr0=1e-3,
        optimizer="adamw",
        pretrained=True,
        verbose=True,
        device="cpu",             # explicitly use CPU
        fraction=1.0,           
        plots=False,            
    )

    print("Training finished (light run).")
    print("Results dir:", results.save_dir)

if __name__ == "__main__":
    main()
