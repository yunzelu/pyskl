import cv2
import csv
import json
import torch
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

def load_processed_data(csv_path, actions_path):
    """Load the pose CSV and the action predictions JSON."""
    # Read Actions
    with open(actions_path, 'r') as f:
        actions_data = json.load(f)
        
    # Read Poses from CSV and group by UnixTime
    poses_by_time = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            unix_time = float(row['UnixTime'])
            poses_by_time[unix_time].append(row)
            
    return poses_by_time, actions_data

def get_action_label(actions_data, track_id, current_unix_time):
    """
    Find the action prediction. If overlapping windows exist, 
    we take the LAST (most recent) one for the most updated context.
    """
    matches = []
    for act in actions_data:
        # Match PersonID and check if current time falls within the window
        if int(act["track_id"]) == track_id and act["start_unix_time"] <= current_unix_time <= act["end_unix_time"]:
            matches.append(act)
            
    if matches:
        latest_act = matches[-1] # Grab the most recent overlapping window
        return f"{latest_act['action']} {latest_act['confidence']:.2f}"
    return "Pending..."

def get_bbox_from_keypoints(keypoints, padding=20):
    """Infer a bounding box by finding the min/max X and Y of valid keypoints."""
    # Filter out keypoints with 0 confidence
    valid_kpts = [kp for kp in keypoints if kp[2] > 0]
    
    if not valid_kpts:
        return [0, 0, 0, 0]
        
    xs = [kp[0] for kp in valid_kpts]
    ys = [kp[1] for kp in valid_kpts]
    
    # Add padding so the box isn't hugging the skeleton too tightly
    x1, y1 = min(xs) - padding, min(ys) - padding
    x2, y2 = max(xs) + padding, max(ys) + padding
    
    return [x1, y1, x2, y2]

def run_csv_visualizer(bg_image_path, video_out, poses_path, actions_path, fps=30):
    poses_by_time, actions_data = load_processed_data(poses_path, actions_path)
    
    # Load the static background image
    bg_frame = cv2.imread(bg_image_path)
    if bg_frame is None:
        raise ValueError(f"Could not load background image from {bg_image_path}")
        
    h, w = bg_frame.shape[:2]
    
    # Setup VideoWriter
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # Sort the unique timestamps so we render the video in chronological order
    sorted_times = sorted(poses_by_time.keys())
    
    print(f"Baking native YOLO-style render to: {video_out}")

    for unix_time in tqdm(sorted_times, desc="Rendering Video Frames", unit="frame"):
        # Start with a fresh copy of the background image for this frame
        frame = bg_frame.copy()
        people = poses_by_time[unix_time]
        
        # Grab the human-readable timestamp from the first person in this frame
        timestamp_str = people[0]['Timestamp']
        
        # Draw the Timestamps on the top left corner
        cv2.putText(frame, f"Time: {timestamp_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Unix: {unix_time}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        annotator = Annotator(frame, line_width=2, example=str("person"))
        
        for p in people:
            tid = int(p["PersonID"])

            if tid == -1:
                continue
            
            # 1. Reconstruct the (17, 3) keypoints array from the CSV row
            kpts = []
            for i in range(17):
                x = float(p[f"KP{i}_X"])
                y = float(p[f"KP{i}_Y"])
                c = float(p[f"KP{i}_C"])
                kpts.append([x, y, c])
            
            # 2. Calculate the bounding box from the keypoints
            box = get_bbox_from_keypoints(kpts, padding=25)
            
            # Convert to tensor for the annotator
            kpts_tensor = torch.tensor(kpts)
            
            # 3. Draw the skeleton
            annotator.kpts(kpts_tensor, shape=(h, w), kpt_line=True)
            
            # 4. Add the Action Recognition label & Bounding Box
            action_label = get_action_label(actions_data, tid, unix_time)
            annotator.box_label(box, f"ID:{tid} | {action_label}", color=colors(tid, True))
        
        # Get the final annotated BGR image and write it
        final_frame = annotator.result()
        out.write(final_frame)
        
    out.release()
    print("Video saved successfully.")

if __name__ == "__main__":
    run_csv_visualizer(
        bg_image_path="pipeline/csv_input/WIN_20260313_15_30_37_Pro.jpg", 
        video_out="pipeline/csv_input/12-12_rendered_v1.1.mp4",
        poses_path="pipeline/csv_input/12-12_padded_corrected.csv",
        actions_path="pipeline/csv_input/12-12_result_v1.1.json",
        fps=30 
    )