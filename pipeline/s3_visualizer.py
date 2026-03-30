import cv2
import json
import torch
import numpy as np
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors

def load_processed_data(poses_path, actions_path):
    """Load the decoupled data files."""
    with open(poses_path, 'r') as f:
        poses_raw = json.load(f)
    poses_by_frame = defaultdict(list)
    for p in poses_raw:
        poses_by_frame[p["frame_id"]].append(p)
        
    with open(actions_path, 'r') as f:
        actions_data = json.load(f)
        
    return poses_by_frame, actions_data

def get_action_label(actions_data, track_id, frame_idx):
    """Find the specific action prediction for a person in the current frame."""
    for act in actions_data:
        if act["track_id"] == track_id and act["start_frame"] <= frame_idx < act["end_frame"]:
            return f"{act['action']} {act['confidence']:.2f}"
    return None

def run_official_visualizer(video_in, video_out, poses_path, actions_path):
    poses_by_frame, actions_data = load_processed_data(poses_path, actions_path)
    
    cap = cv2.VideoCapture(video_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use 'mp4v' or 'avc1' for broad compatibility
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    frame_idx = 0
    print(f"Baking native YOLO-style render to: {video_out}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        people = poses_by_frame.get(frame_idx, [])
        
        # Initialize the official YOLO Annotator for this frame
        # line_width=2 and font_size=None uses YOLO's auto-scaling logic
        annotator = Annotator(frame, line_width=2, example=str("person"))
        
        for p in people:
            tid = p["track_id"]
            # Convert keypoints back to a tensor for the annotator
            # Shape: (17, 3) -> [x, y, conf]
            kpts = torch.tensor(p["keypoints"])
            
            # 1. Draw the native YOLO skeleton
            # This handles the 0.5 confidence threshold and multi-color limbs automatically
            annotator.kpts(kpts, shape=(h, w), kpt_line=True)
            
            # 2. Add the Action Recognition label
            # We fetch the prediction from your trained ST-GCN++ model's output
            action_label = get_action_label(actions_data, tid, frame_idx)
            box = p["box"] 
            annotator.box_label(box, action_label, color=colors(tid, True))
        
        # Get the final annotated BGR image
        final_frame = annotator.result()
        out.write(final_frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Video saved successfully.")

if __name__ == "__main__":
    run_official_visualizer(
        video_in="pipeline/WIN_20260326_15_06_23_Pro.mp4",
        video_out="pipeline/s3_WIN_20260326_15_06_23_Pro.mp4",
        poses_path="pipeline/WIN_20260326_15_06_23_Pro.json",
        actions_path="pipeline/s2_WIN_20260326_15_06_23_Pro.json"
    )