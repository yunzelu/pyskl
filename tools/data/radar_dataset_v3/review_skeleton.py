import os
import json
import cv2
import argparse

# YOLOv8 COCO 17-Keypoint Connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (5, 7), (7, 9),  # Left Arm
    (6, 8), (8, 10), # Right Arm
    (11, 13), (13, 15), # Left Leg
    (12, 14), (14, 16)  # Right Leg
]

def render_skeleton_on_video(json_file, data_dir, clip_index, output_file, vis_threshold=0.25):
    # 1. Load the JSON data
    print(f"Loading skeleton data from {json_file}...")
    with open(json_file, 'r') as f:
        all_data = json.load(f)
        
    if clip_index >= len(all_data) or clip_index < 0:
        print(f"Error: Clip index {clip_index} is out of bounds. The dataset has {len(all_data)} clips.")
        return
        
    clip_data = all_data[clip_index]
    folder = clip_data['folder']
    start_frame = clip_data['clip_start']
    end_frame = clip_data['clip_end']
    target_label = clip_data['target_label']
    
    print(f"\n--- Verifying Clip Index {clip_index} ---")
    print(f"Folder: {folder}")
    print(f"Label: {target_label}")
    print(f"Frames: {start_frame} to {end_frame}")
    
    # 2. Locate and open the video
    # Assuming video is always named 'output_video.avi' based on your previous structure
    video_path = os.path.join(data_dir, folder, 'output_video.avi')
    if not os.path.exists(video_path):
        print(f"Error: Could not find video at {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 3. Setup Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30 # Forcing 30 FPS for viewing as you noted earlier
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # 4. Process frames and draw
    for frame_data in clip_data['frames']:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Video ended before the clip data finished.")
            break
            
        keypoints = frame_data['keypoints']
        scores = frame_data['keypoint_scores']
        bbox = frame_data['bbox']
        det_score = frame_data['det_score']
        frame_idx = frame_data['frame_idx']
        
        # Check if we have a valid detection (bbox is not all zeros)
        if sum(bbox) > 0:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw Bounding Box (Blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Conf: {det_score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw Skeleton Bones (Green)
            for j1, j2 in SKELETON_CONNECTIONS:
                score1, score2 = scores[j1], scores[j2]
                if score1 > vis_threshold and score2 > vis_threshold:
                    pt1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
                    pt2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                    
            # Draw Joints (Green if good, Red if bad)
            for idx, (kp, score) in enumerate(zip(keypoints, scores)):
                x, y = int(kp[0]), int(kp[1])
                if x == 0 and y == 0: 
                    continue # Skip empty points completely
                    
                if score > vis_threshold:
                    # Good point (Green)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                else:
                    # Bad point - ST-GCN++ will zero this out! (Red)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        # Add basic info text to the top corner
        text = f"Frame: {frame_idx} | Label: {target_label}"
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"\nVerification video saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Render YOLO skeleton on a specific clip.")
    parser.add_argument('-j', '--json', type=str, required=True, help="JSON file containing skeleton data.")
    parser.add_argument('-d', '--dir', type=str, default='.', help="Root directory containing the video folders.")
    parser.add_argument('-i', '--index', type=int, default=0, help="Index of the clip in the JSON to verify (0, 1, 2...).")
    parser.add_argument('-o', '--output', type=str, default='verify_output.mp4', help="Output video filename.")
    parser.add_argument('-t', '--threshold', type=float, default=0.25, help="Confidence threshold for coloring joints.")
    
    args = parser.parse_args()
    
    render_skeleton_on_video(args.json, args.dir, args.index, args.output, args.threshold)

if __name__ == "__main__":
    main()