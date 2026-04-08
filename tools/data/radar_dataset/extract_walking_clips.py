import os
import cv2
import pandas as pd

def extract_subject_id(folder_name):
    """Extracts the subject ID from a folder name like '2-saad-walk'."""
    parts = folder_name.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[1:-1])
    return "unknown_subject"

def process_walking(input_root, output_root, max_frames=120):
    stats_clips = 0
    stats_frames = 0
    
    # Ensure the target "Walking" directory exists
    walking_dir = os.path.join(output_root, 'Walking')
    os.makedirs(walking_dir, exist_ok=True)

    for folder_name in os.listdir(input_root):
        # ONLY process folders ending in '-walk'
        if not folder_name.endswith('-walk'):
            continue

        folder_path = os.path.join(input_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        video_path = os.path.join(folder_path, 'output_video.avi')
        csv_path = os.path.join(folder_path, 'frame_labels.csv')

        if not os.path.exists(video_path) or not os.path.exists(csv_path):
            print(f"Skipping {folder_name}: Missing video or CSV.")
            continue

        subject_id = extract_subject_id(folder_name)
        
        # 1. Parse the CSV to get the start and end frames for Walking
        df = pd.read_csv(csv_path)
        start_frame = None
        end_frame = None
        
        for i in range(len(df)):
            label = str(df.iloc[i]['Label']).strip()
            frame = int(df.iloc[i]['Frame'])
            
            if label == 'Walking':
                start_frame = frame
            elif label == 'END':
                end_frame = frame
                break
                
        if start_frame is None or end_frame is None:
            print(f"Skipping {folder_name}: Could not find both 'Walking' and 'END' markers.")
            continue

        # 2. Process video and chunk it
        cap = cv2.VideoCapture(video_path)
        output_fps = 30.0 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 

        current_frame_idx = 0
        chunk_idx = 0
        frames_in_current_chunk = 0
        active_writer = None

        print(f"Chunking {folder_name} (Subject: {subject_id})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # If we reached the END marker, we are done with this folder
            if current_frame_idx >= end_frame:
                if active_writer is not None:
                    active_writer.release()
                    stats_clips += 1
                break

            # If we are within the Walking interval
            if current_frame_idx >= start_frame:
                # Open a new writer if we don't have one
                if active_writer is None:
                    # Naming convention: subject_foldername_chunk0.avi
                    clip_filename = f"{subject_id}_{folder_name}_chunk{chunk_idx}.avi"
                    clip_path = os.path.join(walking_dir, clip_filename)
                    active_writer = cv2.VideoWriter(clip_path, fourcc, output_fps, (width, height))

                # Write frame to the current chunk
                active_writer.write(frame)
                frames_in_current_chunk += 1
                stats_frames += 1

                # If chunk hits the max limit (120 frames), close it
                if frames_in_current_chunk == max_frames:
                    active_writer.release()
                    active_writer = None
                    frames_in_current_chunk = 0
                    chunk_idx += 1
                    stats_clips += 1

            current_frame_idx += 1

        cap.release()

    # 3. Output dataset statistics
    print("\n" + "="*40)
    print("WALKING DATASET PROCESSING STATISTICS")
    print("="*40)
    print(f"Total Folders Processed: {chunk_idx + 1 if chunk_idx > 0 else 0}")
    print(f"Total 120-Frame Clips Generated: {stats_clips}")
    print(f"Total Walking Frames Extracted: {stats_frames}")
    print("="*40)

# ==========================================
# Execution
# ==========================================
input_dataset_dir = r"D:\lu\research\dataset\raw_collection_25-1-11-radar" # Change this
output_dataset_dir = r"D:\lu\project\pyskl\data\radar_dataset"  # Change this

process_walking(input_dataset_dir, output_dataset_dir, max_frames=120)