import cv2
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from ultralytics.utils.plotting import Annotator, colors

def render_skeletons_from_csv(csv_path, image_path, video_out, view_size=(1920, 1080), fps=30):
    # 1. Read the CSV data
    df = pd.read_csv(csv_path)
    
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
    
    print(f"Baking native YOLO-style render on black background to: {video_out}")
    print(f"Configured Canvas Size: {w}x{h}")

    # 3. Iterate through frames with a progress bar
    for _, row in tqdm(unique_timestamps.iterrows(), total=len(unique_timestamps), desc="Rendering frames"):
        # Safely grab the dual timestamps
        timestamp = row.get('Timestamp', '')
        unix_time = row.get('UnixTime', 0.0)
        
        # Create a blank black frame
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
            
            if box is not None and pid != -1:
                annotator.box_label(box, f"ID: {pid}", color=colors(pid, True))
            
            # 5. Draw the native YOLO skeleton
            kpts_tensor = torch.tensor(kpts_array)
            annotator.kpts(kpts_tensor, shape=(h, w), kpt_line=True)
            
        # Get the annotated frame
        final_frame = annotator.result()
        
        # 6. Add Timestamp and UnixTime to the BOTTOM-RIGHT corner
        text_str = f"Time: {timestamp} | Unix: {unix_time}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5   # Scaled down from 1.0
        thickness = 1      # Scaled down from 2
        
        # Calculate the size of the text to offset it from the bottom-right edges
        text_size, _ = cv2.getTextSize(text_str, font, font_scale, thickness)
        text_w, text_h = text_size
        
        # Set a 15-pixel margin from the bottom and left edges
        margin = 15
        text_x = margin
        text_y = h - margin
        
        cv2.putText(final_frame, text_str, (text_x, text_y), font, 
                    font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        
        out.write(final_frame)
        
    out.release()
    print("Video saved successfully.")

def parse_args():
    parser = argparse.ArgumentParser(description="Render YOLO Skeletons without Inference")
    parser.add_argument('--csv-path', type=str, required=True, help="Path to the input CSV data")
    parser.add_argument('--image-path', type=str, required=True, help="Path to the background layout image")
    parser.add_argument('--video-out', type=str, required=True, help="Path to save the output MP4 video")
    parser.add_argument('--width', type=int, default=1280, help="Canvas width (default: 1280)")
    parser.add_argument('--height', type=int, default=720, help="Canvas height (default: 720)")
    parser.add_argument('--fps', type=int, default=30, help="Video frames per second (default: 30)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    render_skeletons_from_csv(
        csv_path=args.csv_path,
        image_path=args.image_path,
        video_out=args.video_out,
        view_size=(args.width, args.height),
        fps=args.fps
    )