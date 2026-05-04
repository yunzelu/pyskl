import cv2
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics.utils.plotting import Annotator, colors

def render_skeletons_from_csv(csv_path, json_path, image_path, video_out, view_size=(1920, 1080), fps=30):
    # 1. Read the CSV data and Inference JSON
    print(f"Loading CSV data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loading Inference JSON from: {json_path}")
    with open(json_path, 'r') as f:
        inference_data = json.load(f)
    
    # Check if the new bounding box columns exist in this CSV
    bbox_cols = ['BBox_X1', 'BBox_Y1', 'BBox_X2', 'BBox_Y2']
    has_bbox_cols = all(col in df.columns for col in bbox_cols)
    
    # Sort by UnixTime to ensure chronological frame rendering
    if 'UnixTime' in df.columns:
        df = df.sort_values(by='UnixTime')
    
    # Group by timestamps to act as our frame sequences
    cols_to_extract = []
    if 'Timestamp' in df.columns: cols_to_extract.append('Timestamp')
    if 'UnixTime' in df.columns: cols_to_extract.append('UnixTime')
    
    unique_timestamps = df[cols_to_extract].drop_duplicates()
    
    # Configuration
    w, h = view_size
    padding = 0  # Internal padding for the bounding box

    bg_original = cv2.imread(image_path)
    if bg_original is None:
        raise FileNotFoundError(f"Could not load background image at {image_path}")
    
    # Resize image to match the video dimensions
    bg_resized = cv2.resize(bg_original, (w, h))
    
    # 2. Setup VideoWriter
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    print(f"Baking native YOLO-style render on background to: {video_out}")
    print(f"Configured Canvas Size: {w}x{h}")

    # 3. Iterate through frames with a progress bar
    for _, row in tqdm(unique_timestamps.iterrows(), total=len(unique_timestamps), desc="Rendering frames"):
        # Safely grab the dual timestamps
        timestamp = row.get('Timestamp', '')
        unix_time = float(row.get('UnixTime', 0.0))
        
        # Create a blank background frame
        frame = bg_resized.copy()
        
        # Initialize YOLO annotator
        annotator = Annotator(frame, line_width=2, example=str("person"))
        
        # Get all people present in this specific frame
        if 'UnixTime' in df.columns:
            people_in_frame = df[df['UnixTime'] == unix_time]
        else:
            people_in_frame = df[df['Timestamp'] == timestamp]
        
        for _, person in people_in_frame.iterrows():
            pid = int(person.get('ID', person.get('PersonID', 0)))
            
            # --- MODIFICATION: Find the action label for this person at this time ---
            # Find all active windows for this person
            active_windows = [
                w for w in inference_data 
                if int(w['track_id']) == pid and w['start_unix_time'] <= unix_time <= w['end_unix_time']
            ]
            
            if active_windows:
                # Sort by start time, so the last element is the newest overlapping window
                active_windows.sort(key=lambda x: x['start_unix_time'])
                current_action = active_windows[-1]['action']
                
                # Extract confidence, defaulting to 0.0 if not found
                confidence = active_windows[-1].get('confidence', 0.0)
                
                # Format label with action and confidence to 2 decimal places
                label_text = f"ID: {pid} | {current_action} ({confidence:.2f})"
            else:
                label_text = f"ID: {pid} | No Action"
            # ------------------------------------------------------------------------

            # Extract keypoints into shape (17, 3) -> [x, y, conf]
            kpts_data = []
            for i in range(17):
                kpts_data.append([
                    person[f'KP{i}_X'],
                    person[f'KP{i}_Y'],
                    person[f'KP{i}_C']
                ])
                
            kpts_array = np.array(kpts_data, dtype=np.float32)
            
            # 4. Compute or Read Bounding Box
            box = None
            
            if has_bbox_cols and not pd.isna(person['BBox_X1']):
                box = [
                    int(person['BBox_X1']),
                    int(person['BBox_Y1']),
                    int(person['BBox_X2']),
                    int(person['BBox_Y2'])
                ]
            else:
                valid_kpts = kpts_array[kpts_array[:, 2] > 0.2]
                
                if len(valid_kpts) > 0:
                    min_x = np.min(valid_kpts[:, 0])
                    min_y = np.min(valid_kpts[:, 1])
                    max_x = np.max(valid_kpts[:, 0])
                    max_y = np.max(valid_kpts[:, 1])
                    
                    box = [
                        max(0, int(min_x - padding)),
                        max(0, int(min_y - padding)),
                        min(w, int(max_x + padding)),
                        min(h, int(max_y + padding))
                    ]
            
            # Apply the new label text to the bounding box
            if box is not None and pid != -1:
                annotator.box_label(box, label_text, color=colors(pid, True))
            
            # 5. Draw the native YOLO skeleton
            kpts_tensor = torch.tensor(kpts_array)
            annotator.kpts(kpts_tensor, shape=(h, w), kpt_line=True)
            
        # Get the annotated frame
        final_frame = annotator.result()
        
        # 6. Add Timestamp and UnixTime to the BOTTOM-RIGHT corner
        text_str = f"Time: {timestamp} | Unix: {unix_time}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5   
        thickness = 1      
        
        text_size, _ = cv2.getTextSize(text_str, font, font_scale, thickness)
        text_w, text_h = text_size
        
        margin = 15
        text_x = margin
        text_y = h - margin
        
        cv2.putText(final_frame, text_str, (text_x, text_y), font, 
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        out.write(final_frame)
        
    out.release()
    print("Video saved successfully.")

if __name__ == "__main__":
    # Make sure to point the new json_path to your raw inference output!
    render_skeletons_from_csv(
        csv_path="pipeline/csv_input/pose_2026-04-18_310_p.csv",
        json_path="pipeline/csv_input/pose_2026-04-18_310_p_r.json",
        image_path="pipeline/csv_input/310_room.jpg",
        video_out="pipeline/csv_input/310_18.mp4",
        view_size=(1280, 720),
        fps=30
    )