import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_ground_truth_labels(gt_data, frame_idx):
    """
    Finds ALL ground truth labels at a specific frame.
    Returns a list of valid actions (e.g., ["Standing", "Walking"]).
    """
    valid_actions = []
    for segment in gt_data:
        if segment["start_frame"] <= frame_idx <= segment["end_frame"]:
            valid_actions.append(segment["action"])
            
    # Return the list of actions, or None if the list is empty
    return valid_actions if valid_actions else None

def evaluate_predictions(predictions_path, gt_path, class_mapping=None, output_image="pipeline/cnntransformer_eval_confusion_matrix.png"):
    preds_data = load_json(predictions_path)
    gt_data = load_json(gt_path)
    
    y_true = []
    y_pred = []
    
    print(f"Loaded {len(preds_data)} sliding window predictions...")
    
    for pred in preds_data:
        center_frame = (pred["start_frame"] + pred["end_frame"]) // 2
        predicted_action = pred["action"]
        
        # Get a LIST of all valid actions happening at this frame
        true_actions = get_ground_truth_labels(gt_data, center_frame)

        # Skip unlabeled frames
        if true_actions is None:
            continue

        # Apply grouping/mapping to the prediction and ALL ground truth labels
        if class_mapping:
            predicted_action = class_mapping.get(predicted_action, predicted_action)
            true_actions = [class_mapping.get(act, act) for act in true_actions]
        
        # --- PERMISSIVE MATCHING LOGIC ---
        if predicted_action in true_actions:
            # The model guessed one of the overlapping actions correctly!
            y_true.append(predicted_action)
        else:
            # The model was wrong. We record the first ground truth action as the "Expected" 
            # label so the confusion matrix logs a clear False Negative/False Positive.
            y_true.append(true_actions[0])
            
        y_pred.append(predicted_action)
        
    print(f"Evaluating {len(y_true)} matched predictions (handled overlaps & skipped unlabeled)...")
    
    if not y_true:
        print("Warning: No predictions matched any ground truth labels!")
        return

    labels = sorted(list(set(y_true + y_pred)))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Label (Output from Model)', fontsize=12, fontweight='bold')
    plt.ylabel('True Label (Ground Truth)', fontsize=12, fontweight='bold')
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
        predictions_path="pipeline/s2_cnntransformer.json",
        gt_path="pipeline/gt_WIN_20260326_15_06_23_Pro.json",
        class_mapping=my_grouping_map
    )