import json
import argparse
import numpy as np

def clean_pad_and_merge(input_json_path, output_json_path, gap_threshold=15, score_thresh=0.5, low_score_limit=8):
    print(f"Loading data from: {input_json_path}")
    with open(input_json_path, 'r') as f:
        dataset_info = json.load(f)

    cleaned_dataset = []
    dropped_clips = 0
    dropped_persons = 0

    for clip in dataset_info:
        clip_name = clip['frame_dir']
        total_frames = clip['total_frames']

        if not clip['persons']:
            print(f"[DROPPED CLIP] '{clip_name}': No persons detected by YOLO.")
            dropped_clips += 1
            continue

        # ==========================================
        # PHASE 1: Track Merging (Fixes ID Switches)
        # ==========================================
        merged_kpts = np.zeros((total_frames, 17, 2))
        merged_scores = np.zeros((total_frames, 17))
        merged_bboxes = np.zeros((total_frames, 4))

        for person in clip['persons']:
            kpts = np.array(person['keypoints'])
            scores = np.array(person['keypoint_scores'])
            bboxes = np.array(person['bboxes'])
            
            valid_mask = np.sum(np.abs(bboxes), axis=1) > 0
            
            merged_kpts[valid_mask] = kpts[valid_mask]
            merged_scores[valid_mask] = scores[valid_mask]
            merged_bboxes[valid_mask] = bboxes[valid_mask]

        clip['persons'] = [{
            'person_id': 1, 
            'keypoints': merged_kpts,
            'keypoint_scores': merged_scores,
            'bboxes': merged_bboxes
        }]

        # ==========================================
        # PHASE 2: Confidence Masking
        # ==========================================
        for person in clip['persons']:
            kpts = person['keypoints']
            scores = person['keypoint_scores']
            bboxes = person['bboxes']
            
            for t in range(total_frames):
                low_score_count = np.sum(scores[t] < score_thresh)
                if low_score_count >= low_score_limit:
                    kpts[t] = 0.0
                    scores[t] = 0.0
                    bboxes[t] = 0.0

        # ==========================================
        # PHASE 3: Edge Trimming & Internal Padding
        # ==========================================
        valid_persons_to_keep = []

        for person in clip['persons']:
            kpts = person['keypoints']
            scores = person['keypoint_scores']
            bboxes = person['bboxes']
            
            valid_mask = np.sum(np.abs(bboxes), axis=1) > 0 
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                print(f"  -> [DROPPED PERSON] '{clip_name}': Lost entirely due to low confidence.")
                dropped_persons += 1
                continue

            first_valid = int(valid_indices[0])  # explicitly cast to int
            last_valid = int(valid_indices[-1])  # explicitly cast to int

            if len(valid_indices) > 1:
                internal_gaps = np.diff(valid_indices) - 1
                max_internal_gap = np.max(internal_gaps)
            else:
                max_internal_gap = 0
            
            if max_internal_gap > gap_threshold:
                print(f"  -> [DROPPED PERSON] '{clip_name}': Internal tracking gap of {max_internal_gap} exceeds threshold.")
                dropped_persons += 1
                continue 

            kpts = kpts[first_valid : last_valid + 1]
            scores = scores[first_valid : last_valid + 1]
            bboxes = bboxes[first_valid : last_valid + 1]
            new_total_frames = last_valid - first_valid + 1
            
            trimmed_valid_mask = np.sum(np.abs(bboxes), axis=1) > 0 

            for t in range(1, new_total_frames):
                if not trimmed_valid_mask[t]:
                    kpts[t] = kpts[t-1]
                    scores[t] = scores[t-1]
                    bboxes[t] = bboxes[t-1]

            person['keypoints'] = kpts.tolist()
            person['keypoint_scores'] = scores.tolist()
            person['bboxes'] = bboxes.tolist()
            valid_persons_to_keep.append(person)

        clip['persons'] = valid_persons_to_keep
        
        if not clip['persons']:
            print(f"[DROPPED CLIP] '{clip_name}': No valid persons remaining after filtering.")
            dropped_clips += 1
        else:
            # FIX: Explicitly cast to Python int to prevent JSON serialization error
            clip['total_frames'] = int(new_total_frames) 
            cleaned_dataset.append(clip)

    with open(output_json_path, 'w') as f:
        json.dump(cleaned_dataset, f, indent=4)
        
    print("\n" + "="*50)
    print("DATASET CLEANING & PADDING RESULTS")
    print("="*50)
    print(f"Original clips: {len(dataset_info)}")
    print(f"Total Persons dropped: {dropped_persons}")
    print(f"Total Clips dropped:   {dropped_clips}")
    print(f"Final Clips kept:      {len(cleaned_dataset)}")
    print(f"Cleaned JSON saved to: {output_json_path}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ID switches, filter low-confidence, trim edges, and pad.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to original raw JSON.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save cleaned JSON.")
    parser.add_argument("-g", "--gap_threshold", type=int, default=15, help="Max gap threshold.")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("-l", "--low_score_limit", type=int, default=8, help="Keypoints below threshold to trigger zeroing.")
    
    args = parser.parse_args()
    clean_pad_and_merge(args.input, args.output, args.gap_threshold, args.score_thresh, args.low_score_limit)