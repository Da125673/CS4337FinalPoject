import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def main():
    # 1. Load saved labels from evaluate_yolo_confusion.py
    #    These files should already exist in the SAME folder as this script.
    y_true = np.loadtxt("y_true.txt", dtype=int)
    y_pred = np.loadtxt("y_pred.txt", dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    # 2. Compute metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro    = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro        = f1_score(y_true, y_pred, average='micro', zero_division=0)

    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # 3. Print results
    print("\n===================== METRICS SUMMARY =====================")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1-score:  {f1_macro:.4f}\n")

    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall:    {recall_micro:.4f}")
    print(f"Micro F1-score:  {f1_micro:.4f}")
    print("===========================================================\n")

    print("=== FULL CLASSIFICATION REPORT ===")
    print(report)

    print("\n=== CONFUSION MATRIX SHAPE ===")
    print(cm.shape)
    print(cm)


if __name__ == "__main__":
    main()
