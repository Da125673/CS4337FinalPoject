# src/convert_voc_to_yolo.py

import os
import glob
import xml.etree.ElementTree as ET
from card_classes import CLASS_MAP


def convert_folder(xml_dir, out_label_dir):
    os.makedirs(out_label_dir, exist_ok=True)

    # Recursively find all XML files
    pattern = os.path.join(xml_dir, "**", "*.xml")
    xml_paths = glob.glob(pattern, recursive=True)
    print(f"Found {len(xml_paths)} XML files in {xml_dir}")

    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # image size
        w = float(root.find("size/width").text)
        h = float(root.find("size/height").text)

        # label file name = image base name (without folder, without extension)
        base = os.path.splitext(os.path.basename(xml_path))[0]
        txt_path = os.path.join(out_label_dir, base + ".txt")

        lines = []

        for obj in root.findall("object"):
            name = obj.find("name").text  # e.g. "QH"
            if name not in CLASS_MAP:
                print(f"[WARN] Unknown class '{name}' in {xml_path}")
                continue
            cls_id = CLASS_MAP[name]

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # convert to YOLO normalized (cx, cy, w, h)
            cx = (xmin + xmax) / 2.0 / w
            cy = (ymin + ymax) / 2.0 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
        else:
            print(f"[INFO] No objects found in {xml_path}, skipping label file.")

        print(f"[OK] {xml_path} -> {txt_path}")


if __name__ == "__main__":
    # Compute project root based on this file's location
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    XML_DIR = os.path.join(ROOT_DIR, "data", "Annotations")
    OUT_LABEL_DIR = os.path.join(ROOT_DIR, "data", "labels")

    print("XML_DIR:", XML_DIR)
    print("OUT_LABEL_DIR:", OUT_LABEL_DIR)

    convert_folder(XML_DIR, OUT_LABEL_DIR)
