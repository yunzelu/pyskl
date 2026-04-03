import csv
import numpy as np

def create_windows(csv_path, window_size=60, stride=30, num_joints=17):
    # Read the CSV data
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        
    # Create a mapping of UnixTime to sequential frame IDs for sliding window logic
    unique_times = sorted(list(set(float(row['UnixTime']) for row in data)))
    time_to_frame = {time: idx for idx, time in enumerate(unique_times)}
    frame_to_time = {idx: time for idx, time in enumerate(unique_times)}
    
    # Group data by track_id (PersonID)
    tracks = {}
    max_frame = len(unique_times) - 1
    
    for row in data:
        tid = row['PersonID']
        # Convert UnixTime to our synthetic sequential frame_id
        unix_time = float(row['UnixTime'])
        fid = time_to_frame[unix_time]
        
        if tid not in tracks:
            tracks[tid] = {}
            
        # Reconstruct the 17x3 keypoints array from the flat CSV columns
        kpts = []
        for i in range(num_joints):
            x = float(row[f'KP{i}_X'])
            y = float(row[f'KP{i}_Y'])
            c = float(row[f'KP{i}_C'])
            kpts.append([x, y, c])
            
        tracks[tid][fid] = kpts
        
    windows = []
    # Create sliding windows
    for start_frame in range(0, max_frame - window_size + 1, stride):
        end_frame = start_frame + window_size
        
        # Process 1 person per window
        for tid, frames_dict in tracks.items():
            # Check if person exists in this window
            valid_frames = [f for f in range(start_frame, end_frame) if f in frames_dict]
            
            # If the person is barely in the window, skip
            if len(valid_frames) < (window_size * 0.5): 
                continue
                
            # Get the exact start and end unix times for this specific person's window
            start_unix_time = frame_to_time[valid_frames[0]]
            end_unix_time = frame_to_time[valid_frames[-1]]
                
            # Shape requirements for PYSKL GCNs: (M, T, V, 2) for coords, (M, T, V) for scores
            keypoint = np.zeros((1, window_size, num_joints, 2), dtype=np.float16)
            keypoint_score = np.zeros((1, window_size, num_joints), dtype=np.float16)
            
            for t_idx, f_idx in enumerate(range(start_frame, end_frame)):
                if f_idx in frames_dict:
                    kpts = np.array(frames_dict[f_idx])
                    keypoint[0, t_idx] = kpts[:, :2]
                    keypoint_score[0, t_idx] = kpts[:, 2]
            
            # Construct the fake_anno dictionary expected by PYSKL
            fake_anno = dict(
                frame_dir='',
                label=-1,
                img_shape=(720, 1280), # Replace with your actual video dimensions
                original_shape=(720, 1280),
                start_index=0,
                modality='Pose',
                total_frames=window_size,
                keypoint=keypoint,
                keypoint_score=keypoint_score
            )
            
            windows.append({
                "track_id": tid,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_unix_time": start_unix_time,
                "end_unix_time": end_unix_time,
                "fake_anno": fake_anno
            })
            
    return windows