import json
import argparse
import numpy as np

def clean_and_pad_dataset(input_json_path, output_json_path, gap_threshold=15, score_thresh=0.5, low_score_limit=8):
    print(f"Loading data from: {input_json_path}")
    with open(input_json_path, 'r') as f:
        dataset_info = json.load(f)

    cleaned_dataset = []
    dropped_count = 0

    for clip in dataset_info:
        clip_name = clip['frame_dir']
        total_frames = clip['total_frames']
        drop_clip = False

        if not clip['persons']:
            print(f"[DROPPED] '{clip_name}': No persons detected.")
            dropped_count += 1
            continue

        # ==========================================
        # PHASE 1: Confidence Masking
        # ==========================================
        for person in clip['persons']:
            kpts = np.array(person['keypoints'])         
            scores = np.array(person['keypoint_scores']) 
            bboxes = np.array(person['bboxes'])          

            # If a frame has >= 8 keypoints below the score threshold, zero it out entirely
            for t in range(total_frames):
                low_score_count = np.sum(scores[t] < score_thresh)
                if low_score_count >= low_score_limit:
                    kpts[t] = 0.0
                    scores[t] = 0.0
                    bboxes[t] = 0.0
            
            # Temporarily store arrays back to the person dictionary
            person['keypoints'] = kpts
            person['keypoint_scores'] = scores
            person['bboxes'] = bboxes

        # ==========================================
        # PHASE 2: Edge Trimming (Cut beginning/end zeros)
        # ==========================================
        # Find the absolute first and last valid frames across all persons in the clip
        first_valid_global = total_frames
        last_valid_global = -1

        for person in clip['persons']:
            valid_mask = np.sum(np.abs(person['bboxes']), axis=1) > 0 
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                first_valid_global = min(first_valid_global, valid_indices[0])
                last_valid_global = max(last_valid_global, valid_indices[-1])

        # If the entire clip became zeros after Phase 1, drop it
        if first_valid_global > last_valid_global:
            print(f"[DROPPED] '{clip_name}': No valid frames remaining after confidence filtering.")
            dropped_count += 1
            continue

        # Trim the clip to physically remove the edge zeros
        new_total_frames = last_valid_global - first_valid_global + 1
        clip['total_frames'] = int(new_total_frames)

        for person in clip['persons']:
            # Slice the arrays to the new temporal boundaries
            person['keypoints'] = person['keypoints'][first_valid_global:last_valid_global + 1]
            person['keypoint_scores'] = person['keypoint_scores'][first_valid_global:last_valid_global + 1]
            person['bboxes'] = person['bboxes'][first_valid_global:last_valid_global + 1]

        # ==========================================
        # PHASE 3: Internal Gap Checking & Padding
        # ==========================================
        for person in clip['persons']:
            kpts = person['keypoints']
            scores = person['keypoint_scores']
            bboxes = person['bboxes']
            
            valid_mask = np.sum(np.abs(bboxes), axis=1) > 0 
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                drop_clip = True
                break

            # 3A: Check maximum continuous gap length
            if len(valid_indices) > 1:
                internal_gaps = np.diff(valid_indices) - 1
                max_internal_gap = np.max(internal_gaps)
            else:
                max_internal_gap = 0
            
            if max_internal_gap > gap_threshold:
                print(f"[DROPPED] '{clip_name}': Continuous tracking loss of {max_internal_gap} frames exceeds threshold.")
                drop_clip = True
                break 

            # 3B: Forward Pad the last valid frame
            # Since we trimmed the edges, index 0 is guaranteed to be a valid frame.
            # We iterate forward. If a frame is invalid, we overwrite it with the previous frame's data.
            for t in range(1, new_total_frames):
                if not valid_mask[t]:
                    kpts[t] = kpts[t-1]
                    scores[t] = scores[t-1]
                    bboxes[t] = bboxes[t-1]
                    # Logically, this frame is now valid for the next iteration's padding

            # Convert back to native Python lists for JSON serialization
            person['keypoints'] = kpts.tolist()
            person['keypoint_scores'] = scores.tolist()
            person['bboxes'] = bboxes.tolist()

        if not drop_clip:
            cleaned_dataset.append(clip)
        else:
            dropped_count += 1

    # Save to the new JSON
    with open(output_json_path, 'w') as f:
        json.dump(cleaned_dataset, f, indent=4)
        
    print("\n" + "="*50)
    print("DATASET CLEANING & PADDING RESULTS")
    print("="*50)
    print(f"Original clips: {len(dataset_info)}")
    print(f"Clips dropped:  {dropped_count}")
    print(f"Clips kept:     {len(cleaned_dataset)}")
    print(f"Cleaned JSON saved to: {output_json_path}")
    print("="*50)

# ==========================================
# Command Line Interface
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter low-confidence poses, trim edges, and pad internal gaps.")
    
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to original raw JSON.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save cleaned JSON.")
    parser.add_argument("-g", "--gap_threshold", type=int, default=15, 
                        help="Max allowed continuous zero frames before clip is dropped. Default: 15")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.5, 
                        help="Confidence score threshold for keypoints. Default: 0.5")
    parser.add_argument("-l", "--low_score_limit", type=int, default=9, 
                        help="Number of keypoints below score_thresh required to zero out the frame. Default: 9")
    
    args = parser.parse_args()
    # output = args.input.split(".")[0] + "_clean.json"
    clean_and_pad_dataset(args.input, args.output, args.gap_threshold, args.score_thresh, args.low_score_limit)