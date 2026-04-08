import os
import cv2
import pandas as pd
from collections import defaultdict

def extract_subject_id(folder_name):
    """
    Extracts the subject ID from a folder name like '2-saad-sit'.
    Assumes the format is [number]-[subject_id]-[activity]
    """
    parts = folder_name.split('-')
    if len(parts) >= 3:
        # Join middle parts in case subject name has a hyphen (e.g., '2-jean-luc-sit')
        return '-'.join(parts[1:-1])
    return "unknown_subject"

def process_radar_dataset(input_root, output_root):
    # Statistics tracking
    stats = defaultdict(lambda: {'clip_count': 0, 'total_frames': 0})
    
    # Ensure output root exists
    os.makedirs(output_root, exist_ok=True)

    # Iterate through all subfolders in the root directory
    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)
        
        if not os.path.isdir(folder_path):
            continue

        video_path = os.path.join(folder_path, 'output_video.avi')
        csv_path = os.path.join(folder_path, 'frame_labels.csv')

        if not os.path.exists(video_path) or not os.path.exists(csv_path):
            print(f"Skipping {folder_name}: Missing video or CSV.")
            continue

        subject_id = extract_subject_id(folder_name)
        
        # 1. Parse the CSV to get target intervals
        df = pd.read_csv(csv_path)
        intervals = [] # List of tuples: (start_frame, end_frame, label)
        
        for i in range(len(df) - 1):
            start_frame = int(df.iloc[i]['Frame'])
            label = str(df.iloc[i]['Label']).strip()
            end_frame = int(df.iloc[i+1]['Frame'])

            if label == 'END':
                break # We reached the end marker, ignore everything after
                
            if label not in ['Walking', 'DELETE']:
                intervals.append((start_frame, end_frame, label))

        # 2. Process the video sequentially to bypass the 20fps/30fps header issue
        cap = cv2.VideoCapture(video_path)
        
        # We explicitly set our output writers to 30 FPS
        output_fps = 30.0 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # You can change to 'mp4v' if you prefer .mp4

        current_frame_idx = 0
        interval_idx = 0
        active_writer = None
        current_label = None
        
        print(f"Processing {folder_name} (Subject: {subject_id})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # If we've processed all our intervals, we can stop reading this video early
            if interval_idx >= len(intervals) and active_writer is None:
                break

            # Check if we need to close an active writer (we reached the end_frame)
            if active_writer is not None:
                _, end_frame, _ = intervals[interval_idx]
                if current_frame_idx >= end_frame:
                    active_writer.release()
                    active_writer = None
                    interval_idx += 1
                    
                    # Re-check if we are done with all intervals after incrementing
                    if interval_idx >= len(intervals):
                        break

            # Check if we need to open a new writer (we reached the start_frame of the next interval)
            if active_writer is None and interval_idx < len(intervals):
                start_frame, _, label = intervals[interval_idx]
                if current_frame_idx == start_frame:
                    current_label = label
                    
                    # Create directory for this label
                    label_dir = os.path.join(output_root, current_label)
                    os.makedirs(label_dir, exist_ok=True)
                    
                    # Generate a unique, traceable filename
                    clip_filename = f"{subject_id}_{folder_name}_{start_frame}.avi"
                    clip_path = os.path.join(label_dir, clip_filename)
                    
                    active_writer = cv2.VideoWriter(clip_path, fourcc, output_fps, (width, height))
                    stats[current_label]['clip_count'] += 1

            # If we are inside an interval, write the frame and update stats
            if active_writer is not None:
                active_writer.write(frame)
                stats[current_label]['total_frames'] += 1

            current_frame_idx += 1

        # Clean up in case video ended before the interval closed
        if active_writer is not None:
            active_writer.release()
            
        cap.release()

    # 3. Output dataset statistics
    print("\n" + "="*40)
    print("DATASET PROCESSING STATISTICS")
    print("="*40)
    total_clips = 0
    total_frames = 0
    
    for label, stat in sorted(stats.items()):
        clips = stat['clip_count']
        frames = stat['total_frames']
        total_clips += clips
        total_frames += frames
        print(f"{label}:")
        print(f"  - Clips extracted: {clips}")
        print(f"  - Total frames: {frames}")
        print(f"  - Avg frames/clip: {frames/clips:.1f}" if clips > 0 else "")
        
    print("-" * 40)
    print(f"TOTAL NON-WALKING CLIPS: {total_clips}")
    print(f"TOTAL NON-WALKING FRAMES: {total_frames}")
    print("="*40)

# ==========================================
# Execution
# ==========================================
input_dataset_dir = r"D:\lu\research\dataset\raw_collection_25-1-11-radar" # Change this
output_dataset_dir = r"D:\lu\project\pyskl\data\radar_dataset"  # Change this

process_radar_dataset(input_dataset_dir, output_dataset_dir)