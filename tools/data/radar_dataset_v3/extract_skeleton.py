import os
import csv
import cv2
import json
import argparse
from collections import Counter
from ultralytics import YOLO

def extract_subject_id(folder_name):
    parts = folder_name.split('-')
    if len(parts) >= 2:
        return parts[1]
    return "unknown"

def process_clips(csv_file, data_dir, model_path, tracker_config, device, output_file):
    print(f"Loading YOLO model from {model_path} on device: {device}...")
    model = YOLO(model_path)
    
    clips = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append(row)
            
    all_skeleton_data = []
    subject_counter = Counter()
    class_counter = Counter()
    
    total_frames_processed = 0
    total_frames_dropped = 0
    total_clips_dropped = 0 # NEW: Counter for whole clips dropped

    print(f"Starting extraction for {len(clips)} clips...")
    
    for idx, clip in enumerate(clips):
        folder = clip['Folder']
        video_file = clip['VideoFile']
        start_frame = int(clip['ClipStart'])
        end_frame = int(clip['ClipEnd'])
        target_label = clip['TargetLabel']
        
        video_path = os.path.join(data_dir, folder, video_file)
        if not os.path.exists(video_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        subject_id = extract_subject_id(folder)
        
        clip_data = {
            "folder": folder,
            "subject_id": subject_id,
            "target_label": target_label,
            "clip_start": start_frame,
            "clip_end": end_frame,
            "frames": [] # We will only append valid frames here
        }
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames_processed += 1
            
            results = model.track(
                frame, 
                persist=True, 
                verbose=False, 
                imgsz=640,
                conf=0.1,  
                tracker=tracker_config,
                device=device
            )
            
            # Check if anyone was detected
            if results and len(results[0].boxes) > 0:
                # Strictly pick the highest confidence box
                best_idx = results[0].boxes.conf.argmax().item()
                
                box = results[0].boxes[best_idx]
                keypoints = results[0].keypoints[best_idx]
                
                frame_result = {
                    "frame_idx": frame_idx,
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "det_score": float(box.conf[0].cpu().numpy()),
                    "keypoints": [],
                    "keypoint_scores": []
                }
                
                if keypoints.has_visible:
                    frame_result["keypoints"] = keypoints.xy[0].cpu().numpy().tolist()
                    frame_result["keypoint_scores"] = keypoints.conf[0].cpu().numpy().tolist()
                
                # ONLY append if a person was found
                clip_data["frames"].append(frame_result)
            else:
                # No person detected. Drop the frame.
                total_frames_dropped += 1
                
        cap.release()
        
        # Calculate actual saved length
        actual_length = len(clip_data["frames"])
        
        # NEW LOGIC: Check if the clip is now too short
        if actual_length < 90:
            print(f"Dropped Clip: {folder} ({start_frame}-{end_frame}). Only {actual_length} valid frames remained after YOLO.")
            total_clips_dropped += 1
            continue # Skip saving this clip and move to the next one
            
        # If it survived, save it and update counters
        clip_data["saved_length"] = actual_length
        all_skeleton_data.append(clip_data)
        
        subject_counter[subject_id] += 1
        class_counter[target_label] += 1
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(clips)} clips...")

    with open(output_file, 'w') as f:
        json.dump(all_skeleton_data, f)
        
    print(f"\n--- DATASET STATISTICS ---")
    print(f"Total Initial Clips: {len(clips)}")
    print(f"Clips Dropped (<90 frames): {total_clips_dropped}")
    print(f"Total Valid Clips Saved: {len(all_skeleton_data)}")
    print(f"Total Video Frames Read: {total_frames_processed}")
    print(f"Empty Frames Dropped: {total_frames_dropped} ({(total_frames_dropped/total_frames_processed)*100:.2f}%)")
    print(f"Data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLO Pose Tracking and drop empty frames/short clips.")
    parser.add_argument('-c', '--csv', type=str, required=True, help="Input CSV file from the previous step.")
    parser.add_argument('-d', '--dir', type=str, default='.', help="Root directory containing the video folders.")
    parser.add_argument('-m', '--model', type=str, default='yolov8x-pose.pt', help="Path to YOLO pose model weights.")
    parser.add_argument('-t', '--tracker', type=str, default='bytetrack.yaml', help="Tracker config.")
    parser.add_argument('--device', type=str, default='0', help="Device: '0' for GPU, 'cpu' for CPU.")
    parser.add_argument('-o', '--output', type=str, default='dataset_skeletons.json', help="Output JSON filename.")
    
    args = parser.parse_args()
    process_clips(args.csv, args.dir, args.model, args.tracker, args.device, args.output)

if __name__ == "__main__":
    main()