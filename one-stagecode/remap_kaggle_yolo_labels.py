# src/remap_kaggle_yolo_labels.py

import os
import glob
from card_classes import CLASS_MAP

def alpha_prefix(s: str) -> str:
    """Return leading letters from a string, e.g. 'KS4' -> 'KS'."""
    out = []
    for c in s:
        if c.isalpha():
            out.append(c.upper())
        else:
            break
    return "".join(out)

def remap_folder(label_dir):
    txt_paths = glob.glob(os.path.join(label_dir, "*.txt"))
    print(f"Found {len(txt_paths)} YOLO label files in {label_dir}")

    for txt_path in txt_paths:
        base = os.path.splitext(os.path.basename(txt_path))[0]  # e.g. "KS4"
        code = alpha_prefix(base)                               # "KS"

        if code not in CLASS_MAP:
            print(f"[WARN] Unknown code '{code}' from {base}, skipping")
            continue

        cls_id = CLASS_MAP[code]
        new_lines = []

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                # ignore old class id, keep bbox
                _, cx, cy, w, h = parts
                new_lines.append(f"{cls_id} {cx} {cy} {w} {h}")

        with open(txt_path, "w") as f:
            f.write("\n".join(new_lines))

        print(f"[OK] Remapped {txt_path} to class id {cls_id} ({code})")

if __name__ == "__main__":
    # TODO: point this at your Kaggle TXT label folder(s)
    LABEL_DIR = "../data/labels/train"
    remap_folder(LABEL_DIR)
