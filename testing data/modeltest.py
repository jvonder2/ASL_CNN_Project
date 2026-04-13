"""
ASL Model Evaluation Script
===========================
Evaluates a trained Keras ASL classifier on a labeled image dataset.

Folder structure expected:
    DATA_DIR/
        A/
        B/
        C/
        ...
        Z/
        del/
        space/

Outputs:
- overall accuracy
- macro precision / recall / F1
- weighted precision / recall / F1
- full classification report
- confusion matrix image
- JSON + text report files
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = r"C:\Users\jvond\ML_Project\third model\model\thirdmodel.keras"
DATA_DIR = r"C:\Users\jvond\ML_Project\testing data\test_photos"
SAVE_DIR = r"C:\Users\jvond\ML_Project\third model\third dataset"

IMG_SIZE = 96
BATCH_SIZE = 16

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# LOAD DATASET
# =========================================================
print("Loading dataset...")

dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,          # IMPORTANT for correct label alignment
    label_mode="int",
)

class_names = dataset.class_names
num_classes = len(class_names)

print(f"Found {num_classes} classes:")
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# =========================================================
# LOAD MODEL
# =========================================================
print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded:", MODEL_PATH)

# =========================================================
# RUN PREDICTIONS
# =========================================================
print("\nRunning predictions...")

y_true = []
y_pred = []

for images, labels in dataset:
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(pred_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================================================
# METRICS
# =========================================================
overall_accuracy = accuracy_score(y_true, y_pred)

macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0,
)

# =========================================================
# PRINT SUMMARY
# =========================================================
print("\n===== OVERALL METRICS =====")
print(f"Accuracy:           {overall_accuracy:.4f}")
print(f"Macro Precision:    {macro_precision:.4f}")
print(f"Macro Recall:       {macro_recall:.4f}")
print(f"Macro F1-score:     {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall:    {weighted_recall:.4f}")
print(f"Weighted F1-score:  {weighted_f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(report_text)

# =========================================================
# SAVE METRICS
# =========================================================
summary = {
    "accuracy": overall_accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_precision": weighted_precision,
    "weighted_recall": weighted_recall,
    "weighted_f1": weighted_f1,
    "num_classes": num_classes,
    "class_names": class_names,
}

with open(os.path.join(SAVE_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(SAVE_DIR, "classification_report.json"), "w") as f:
    json.dump(report_dict, f, indent=2)

with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
    f.write(report_text)

print(f"\nSaved reports to: {SAVE_DIR}")

# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Confusion Matrix")
plt.colorbar(im)

tick_marks = np.arange(num_classes)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(class_names, rotation=90, fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)

ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")

plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()

print(f"Saved confusion matrix to: {cm_path}")