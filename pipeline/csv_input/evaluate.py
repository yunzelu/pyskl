import json
import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score

# ---------------------------------------------------------
# 1. Helper Functions to Convert Data
# ---------------------------------------------------------
def json_to_timeline(data_list, start_time, end_time, fps=10):
    """
    Converts JSON time segments into a 1D array (timeline) of 0.1s bins.
    """
    total_frames = int(np.ceil((end_time - start_time) * fps))
    # Default label is "Background" if empty
    timeline = np.full(total_frames, "Background", dtype=object)
    
    for item in data_list:
        # Check for both Ground Truth and Inference JSON keys
        t_start = item.get('start_unix_time', item.get('start_time'))
        t_end = item.get('end_unix_time', item.get('end_time'))
        label = item.get('action', item.get('label'))
        
        # Calculate array indices for 0.1s bins
        idx_start = max(0, int((t_start - start_time) * fps))
        idx_end = min(total_frames, int((t_end - start_time) * fps))
        
        # Apply Rule 1: Combine Lying classes
        if label in ["LayBed-Stationary", "LayFloor-Stationary"]:
            # label = "Lying"
            label = "Lay-Stationary"

            
        timeline[idx_start:idx_end] = label
        
    return timeline

def timeline_to_segments(timeline):
    """
    Groups the 1D array back into segments for F1@k evaluation.
    """
    segments = []
    if len(timeline) == 0:
        return segments
        
    current_label = timeline[0]
    start_idx = 0
    
    for i in range(1, len(timeline)):
        if timeline[i] != current_label:
            segments.append({'label': current_label, 'start_idx': start_idx, 'end_idx': i})
            current_label = timeline[i]
            start_idx = i
            
    segments.append({'label': current_label, 'start_idx': start_idx, 'end_idx': len(timeline)})
    return segments

# ---------------------------------------------------------
# 2. Metric Calculations
# ---------------------------------------------------------
def calculate_f1_at_k(gt_segments, pred_segments, k_threshold=0.5):
    """
    Calculates Segmental F1@k using Intersection over Union (IoU).
    """
    true_positives = 0
    false_positives = 0
    false_negatives = len(gt_segments)
    
    matched_gt = set()
    
    for pred in pred_segments:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_segments):
            if pred['label'] == gt['label'] and i not in matched_gt:
                intersection = max(0, min(pred['end_idx'], gt['end_idx']) - max(pred['start_idx'], gt['start_idx']))
                union = (pred['end_idx'] - pred['start_idx']) + (gt['end_idx'] - gt['start_idx']) - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    
        if best_iou >= k_threshold:
            true_positives += 1
            false_negatives -= 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1
            
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# ---------------------------------------------------------
# 3. Main Evaluation Pipeline
# ---------------------------------------------------------
def evaluate_invisiguard(gt_json, pred_json):
    # Find the global start and end times to align both timelines
    all_times = [item.get('start_unix_time', item.get('start_time')) for item in gt_json + pred_json] + \
                [item.get('end_unix_time', item.get('end_time')) for item in gt_json + pred_json]
    
    start_time = min(all_times)
    end_time = max(all_times)
    
    # Create aligned timelines (10 FPS = 0.1s bins)
    gt_timeline = json_to_timeline(gt_json, start_time, end_time, fps=10)
    pred_timeline = json_to_timeline(pred_json, start_time, end_time, fps=10)
    
    # =========================================================
    # MODEL-LEVEL EVALUATION (Skipping specific periods)
    # =========================================================
    # Rule 2: Create a True/False mask to skip bad periods
    valid_mask = (gt_timeline != "Out-of-Room") & \
                 (gt_timeline != "Multiperson") & \
                 (pred_timeline != "Multiperson") & \
                 (pred_timeline != "No-Detection") & \
                 (pred_timeline != "Background")
                 
    # Filter timelines using the mask
    gt_model_level = gt_timeline[valid_mask]
    pred_model_level = pred_timeline[valid_mask]
    
    # Convert back to segments
    gt_segments_model = timeline_to_segments(gt_model_level)
    pred_segments_model = timeline_to_segments(pred_model_level)
    
    # Calculate Metrics
    model_f1_10 = calculate_f1_at_k(gt_segments_model, pred_segments_model, k_threshold=0.10)
    model_f1_50 = calculate_f1_at_k(gt_segments_model, pred_segments_model, k_threshold=0.50)
    
    labels_model = list(set(gt_model_level))
    model_macro_recall = recall_score(gt_model_level, pred_model_level, average='macro', labels=labels_model, zero_division=0)
    model_macro_f1 = f1_score(gt_model_level, pred_model_level, average='macro', labels=labels_model, zero_division=0)
    model_micro_acc = accuracy_score(gt_model_level, pred_model_level)
    
    print("\n" + "="*40)
    print("MODEL LEVEL EVALUATION")
    print("="*40)
    print(f"Segmental F1@10:         {model_f1_10:.4f}")
    print(f"Segmental F1@50:         {model_f1_50:.4f}")
    print(f"Frame-wise Macro Recall: {model_macro_recall:.4f}")
    print(f"Frame-wise Macro F1:     {model_macro_f1:.4f}")
    print(f"Frame-wise Micro Acc:    {model_micro_acc:.4f}")
    
    # =========================================================
    # SYSTEM-LEVEL EVALUATION (No skipping, Map No-Detection)
    # =========================================================
    pred_sys_timeline = pred_timeline.copy()
    
    # Rule 3: Map "No-Detection" and empty space to "Out-of-Room"
    pred_sys_timeline[pred_sys_timeline == "No-Detection"] = "Out-of-Room"
    pred_sys_timeline[pred_sys_timeline == "Background"] = "Out-of-Room" 
    
    gt_segments_sys = timeline_to_segments(gt_timeline)
    pred_segments_sys = timeline_to_segments(pred_sys_timeline)
    
    # Calculate Metrics
    sys_f1_10 = calculate_f1_at_k(gt_segments_sys, pred_segments_sys, k_threshold=0.10)
    sys_f1_50 = calculate_f1_at_k(gt_segments_sys, pred_segments_sys, k_threshold=0.50)
    
    labels_sys = list(set(gt_timeline))
    sys_macro_recall = recall_score(gt_timeline, pred_sys_timeline, average='macro', labels=labels_sys, zero_division=0)
    sys_macro_f1 = f1_score(gt_timeline, pred_sys_timeline, average='macro', labels=labels_sys, zero_division=0)
    sys_micro_acc = accuracy_score(gt_timeline, pred_sys_timeline)
    
    print("\n" + "="*40)
    print("SYSTEM LEVEL EVALUATION")
    print("="*40)
    print(f"Segmental F1@10:         {sys_f1_10:.4f}")
    print(f"Segmental F1@50:         {sys_f1_50:.4f}")
    print(f"Frame-wise Macro Recall: {sys_macro_recall:.4f}")
    print(f"Frame-wise Macro F1:     {sys_macro_f1:.4f}")
    print(f"Frame-wise Micro Acc:    {sys_micro_acc:.4f}\n")

# ---------------------------------------------------------
# 4. File Loading and Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    GT_FILE_PATH = "/mnt/d/lu/project/auto_labeling_pipeline/data/Willowbend/310/gt_pose_2026-04-18_310.json"
    PRED_FILE_PATH = "/mnt/d/lu/project/auto_labeling_pipeline/data/Willowbend/310/pose_2026-04-18_310_p_r2_f.json"
    
    try:
        print(f"Loading Ground Truth from: {GT_FILE_PATH}")
        with open(GT_FILE_PATH, 'r') as file:
            gt_data = json.load(file)
            
        print(f"Loading Inference from:    {PRED_FILE_PATH}")
        with open(PRED_FILE_PATH, 'r') as file:
            pred_data = json.load(file)
            
        print("Starting Evaluation Pipeline...")
        evaluate_invisiguard(gt_data, pred_data)
        
    except FileNotFoundError as e:
        print(f"\nError: Could not find the JSON file. Please check the file path.\nDetails: {e}")
    except json.JSONDecodeError:
        print("\nError: The file is not a valid JSON. Please check the file contents.")