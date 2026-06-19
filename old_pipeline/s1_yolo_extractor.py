from ultralytics import YOLO
import json
import numpy as np

def extract_and_track(video_path, output_json):
    # Load a YOLOv8-pose model
    model = YOLO('checkpoints/yolo26n-pose.pt')
    
    # Run tracking on the video
    results = model.track(source=video_path, show=False, tracker="botsort.yaml", stream=True)
    
    video_data = []
    for frame_idx, r in enumerate(results):
        if r.boxes.id is None:
            continue # No track IDs in this frame
            
        track_ids = r.boxes.id.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        keypoints = r.keypoints.data.cpu().numpy() # Shape: (Num_persons, 17, 3)
        
        for i, track_id in enumerate(track_ids):
            # keypoints[i] contains [x, y, confidence] for 17 joints
            person_data = {
                "frame_id": frame_idx,
                "track_id": int(track_id),
                "box": boxes[i].tolist(),
                "keypoints": keypoints[i].tolist() 
            }
            video_data.append(person_data)

    with open(output_json, 'w') as f:
        json.dump(video_data, f)

if __name__ == "__main__":
    extract_and_track("pipeline/WIN_20260326_15_06_23_Pro.mp4", "pipeline/WIN_20260326_15_06_23_Pro.json")