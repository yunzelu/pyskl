import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_ground_truth_label(gt_data, frame_idx):
    """Finds the ground truth label at a specific frame, ignoring track_id."""
    for segment in gt_data:
        if segment["start_frame"] <= frame_idx <= segment["end_frame"]:
            return segment["action"]
    return None 

def evaluate_predictions(predictions_path, gt_path, class_mapping=None, output_image="confusion_matrix.png"):
    preds_data = load_json(predictions_path)
    gt_data = load_json(gt_path)
    
    y_true = []
    y_pred = []
    
    print(f"Loaded {len(preds_data)} sliding window predictions...")
    
    for pred in preds_data:
        center_frame = (pred["start_frame"] + pred["end_frame"]) // 2
        predicted_action = pred["action"]
        
        true_action = get_ground_truth_label(gt_data, center_frame)

        if true_action is None:
            continue

        if class_mapping:
            predicted_action = class_mapping.get(predicted_action, predicted_action)
            true_action = class_mapping.get(true_action, true_action)
        
        y_true.append(true_action)
        y_pred.append(predicted_action)
        
    print(f"Evaluating {len(y_true)} matched predictions (skipped unlabeled frames)...")
    
    if not y_true:
        print("Warning: No predictions matched any ground truth labels! Check your frame indices.")
        return

    labels = sorted(list(set(y_true + y_pred)))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print("\n--- Confusion Matrix (Console) ---")
    row_format = "{:>15}" * (len(labels) + 1)
    print(row_format.format("", *labels))
    for action, row in zip(labels, cm):
        print(row_format.format(action, *row))

    # --- NEW: Plotting Logic ---
    plt.figure(figsize=(10, 8))
    # annot=True displays the numbers, fmt='d' formats them as integers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=labels, yticklabels=labels)
    
    # Explicitly define which axis is which
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Action Recognition Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"\n[SUCCESS] Confusion matrix plot saved to: {output_image}")

if __name__ == "__main__":
    my_grouping_map = {
        "Standing": "Standing/Walking",
        "Walking": "Standing/Walking"
    }

    evaluate_predictions(
        predictions_path="pipeline/s2_WIN_20260326_15_06_23_Pro.json",
        gt_path="pipeline/gt_WIN_20260326_15_06_23_Pro.json",
        class_mapping=my_grouping_map,
        output_image="pipeline/eval_confusion_matrix.png"
    )