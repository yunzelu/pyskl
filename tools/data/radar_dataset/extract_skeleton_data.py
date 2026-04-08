import os
import cv2
import json
import argparse
import numpy as np
from ultralytics import YOLO

def process_label_folder(input_folder, label_name, output_json_path, weight_path="yolo26.pt", device=0):
    """
    Runs YOLO pose estimation with tracking on all clips in a specific label folder.
    Saves the output to a JSON file structured for easy conversion to PYSKL format.
    """
    print(f"Loading YOLO model from: {weight_path} on device: {device}...")
    model = YOLO(weight_path)
    
    dataset_info = []
    
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.avi', '.mp4'))]
    print(f"Found {len(video_files)} clips for label '{label_name}'.")

    for video_idx, video_file in enumerate(video_files):
        video_path = os.path.join(input_folder, video_file)
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing [{video_idx+1}/{len(video_files)}]: {video_file}")
        
        # Dictionary to store tracking data: { track_id: { 'kpts': {}, 'scores': {}, 'bboxes': {} } }
        tracks_data = {}
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO with tracking, restrict to class 0 (person), force device
            results = model.track(
                frame, 
                persist=True, 
                classes=[0], 
                verbose=False, 
                tracker="bytetrack.yaml",
                device=device
            )
            
            # Check if any persons were detected and tracked
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                kpts = results[0].keypoints.xy.cpu().numpy()     # Shape: [N, V, 2]
                confs = results[0].keypoints.conf.cpu().numpy()  # Shape: [N, V]
                
                for box, tid, kpt, conf in zip(boxes, track_ids, kpts, confs):
                    if tid not in tracks_data:
                        tracks_data[tid] = {'kpts': {}, 'scores': {}, 'bboxes': {}}
                        
                    # Save the data for this specific frame index
                    tracks_data[tid]['kpts'][frame_idx] = kpt.tolist()
                    tracks_data[tid]['scores'][frame_idx] = conf.tolist()
                    tracks_data[tid]['bboxes'][frame_idx] = box.tolist()
                    
            frame_idx += 1
            
        cap.release()
        
        total_frames = frame_idx 
        
        num_kpts = 17 
        if tracks_data:
            first_tid = list(tracks_data.keys())[0]
            first_frame_with_data = list(tracks_data[first_tid]['kpts'].keys())[0]
            num_kpts = len(tracks_data[first_tid]['kpts'][first_frame_with_data])

        persons = []
        for tid, data in tracks_data.items():
            person_kpts = []
            person_scores = []
            person_bboxes = []
            
            for i in range(total_frames):
                if i in data['kpts']:
                    person_kpts.append(data['kpts'][i])
                    person_scores.append(data['scores'][i])
                    person_bboxes.append(data['bboxes'][i])
                else:
                    person_kpts.append([[0.0, 0.0]] * num_kpts)
                    person_scores.append([0.0] * num_kpts)
                    person_bboxes.append([0.0, 0.0, 0.0, 0.0])
                    
            persons.append({
                "person_id": int(tid), # Cast to int natively for clean JSON
                "keypoints": person_kpts,
                "keypoint_scores": person_scores,
                "bboxes": person_bboxes
            })

        dataset_info.append({
            "frame_dir": video_file,
            "label": label_name,
            "total_frames": total_frames,
            "img_shape": [height, width],
            "original_shape": [height, width],
            "persons": persons
        })

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
        
    print(f"\nSaved tracking data for {len(dataset_info)} clips to {output_json_path}")

# ==========================================
# Command Line Interface
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract YOLO skeletons and save to PYSKL-compatible JSON.")
    
    # Required arguments
    parser.add_argument("-i", "--input_folder", type=str, required=True, 
                        help="Path to the folder containing video clips (e.g., .../Transition-Stand-to-Sit).")
    parser.add_argument("-l", "--label", type=str, required=True, 
                        help="The action label name for these clips (e.g., Transition-Stand-to-Sit).")
    parser.add_argument("-o", "--output_file", type=str, required=True, 
                        help="Path to the output JSON file.")
    
    # Optional arguments
    parser.add_argument("-w", "--weight", type=str, default="checkpoints/yolo26x-pose.pt", 
                        help="Path to the YOLO model weights. Default: checkpoints/yolo26x-pose.pt")
    parser.add_argument("-d", "--device", type=str, default="cpu", 
                        help="GPU device ID (e.g., '0' or '1') or 'cpu'. Default: cpu")
    
    args = parser.parse_args()
    
    # Handle passing numeric IDs (like 0) vs strings (like 'cpu') to the device parameter safely
    device_val = int(args.device) if args.device.isdigit() else args.device

    process_label_folder(
        input_folder=args.input_folder, 
        label_name=args.label, 
        output_json_path=args.output_file,
        weight_path=args.weight,
        device=device_val
    )