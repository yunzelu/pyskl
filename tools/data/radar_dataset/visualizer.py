import os
import cv2
import json
import argparse

# COCO 17-keypoint skeleton connection pairs
SKELETON_PAIRS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), # Legs and Pelvis
    (5, 11), (6, 12), (5, 6),                         # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),                  # Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),           # Face
    (3, 5), (4, 6)                                    # Shoulders to Ears
]

def visualize_dataset(json_path, video_dir):
    print(f"Loading JSON data from: {json_path}")
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)

    print("\n--- Playback Controls ---")
    print("[SPACE] : Pause / Play")
    print("[ n ]   : Skip to next clip")
    print("[ q ]   : Quit visualizer")
    print("-------------------------\n")

    for clip in dataset_info:
        clip_name = clip['frame_dir']
        video_path = os.path.join(video_dir, clip_name)

        if not os.path.exists(video_path):
            print(f"[ERROR] Could not find video: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = clip['total_frames']
        
        print(f"Playing: {clip_name} ({total_frames} frames)")
        
        frame_idx = 0
        paused = False
        skip_clip = False

        while cap.isOpened() and frame_idx < total_frames:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw data for every tracked person in this specific frame
                for person in clip['persons']:
                    kpts = person['keypoints'][frame_idx]
                    bbox = person['bboxes'][frame_idx]
                    
                    # 1. Draw Bounding Box (Green)
                    x1, y1, x2, y2 = map(int, bbox)
                    if x1 != 0 or y1 != 0 or x2 != 0 or y2 != 0: # Ensure it's valid
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {person['person_id']}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 2. Draw Skeleton Bones (Blue)
                    for p1, p2 in SKELETON_PAIRS:
                        pt1 = kpts[p1]
                        pt2 = kpts[p2]
                        # Only draw the bone if both connected keypoints are valid (not 0.0)
                        if (pt1[0] != 0.0 and pt1[1] != 0.0) and (pt2[0] != 0.0 and pt2[1] != 0.0):
                            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)

                    # 3. Draw Keypoint Joints (Red)
                    for kpt_idx, (kx, ky) in enumerate(kpts):
                        if kx != 0.0 and ky != 0.0:
                            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)

                cv2.imshow("Skeleton Visualizer", frame)
                frame_idx += 1

            # Handle Keyboard Inputs
            key = cv2.waitKey(30) & 0xFF # ~30ms delay matches 30fps roughly
            
            if key == ord('q'):
                print("Quitting visualizer...")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                print("Skipping to next clip...")
                skip_clip = True
                break
            elif key == ord(' '):
                paused = not paused

        cap.release()
        if skip_clip:
            continue

    cv2.destroyAllWindows()
    print("Finished reviewing all clips in JSON.")

# ==========================================
# Command Line Interface
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO pose JSON data overlaid on the original video clips.")
    
    parser.add_argument("-j", "--json", type=str, required=True, 
                        help="Path to the cleaned JSON file.")
    parser.add_argument("-v", "--video_dir", type=str, required=True, 
                        help="Path to the directory containing the original .avi/.mp4 clips.")
    
    args = parser.parse_args()
    visualize_dataset(args.json, args.video_dir)