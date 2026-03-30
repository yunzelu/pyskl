import json
import numpy as np

def create_windows(json_path, window_size=60, stride=30, num_joints=17):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Group data by track_id
    tracks = {}
    max_frame = 0
    for entry in data:
        tid = entry['track_id']
        fid = entry['frame_id']
        max_frame = max(max_frame, fid)
        if tid not in tracks:
            tracks[tid] = {}
        tracks[tid][fid] = entry['keypoints']
        
    windows = []
    # Create sliding windows
    for start_frame in range(0, max_frame - window_size + 1, stride):
        end_frame = start_frame + window_size
        
        # For simplicity, let's say we process 1 person per window for auto-labeling
        for tid, frames_dict in tracks.items():
            # Check if person exists in this window
            valid_frames = [f for f in range(start_frame, end_frame) if f in frames_dict]
            
            # If the person is barely in the window, skip
            if len(valid_frames) < (window_size * 0.5): 
                continue
                
            # Shape requirements for PYSKL GCNs: (M, T, V, 2) for coords, (M, T, V) for scores
            # M = max_tracks (we'll use 1 here), T = window_size, V = 17
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
                img_shape=(1080, 1920), # Replace with your actual video dimensions
                original_shape=(1080, 1920),
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
                "fake_anno": fake_anno
            })
            
    return windows