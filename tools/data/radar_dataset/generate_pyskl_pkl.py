import os
import json
import pickle
import random
import argparse
import numpy as np

def generate_pyskl_pkl(json_folder, output_pkl, seed=42):
    print(f"Scanning folder for JSON files: {json_folder}")
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    if not json_files:
        print("[ERROR] No JSON files found in the directory.")
        return

    all_clips = []
    unique_labels = set()
    unique_subjects = set()

    # 1. Load all JSON files and collect data
    for j_file in json_files:
        with open(os.path.join(json_folder, j_file), 'r') as f:
            data = json.load(f)
            all_clips.extend(data)
            
            for clip in data:
                unique_labels.add(clip['label'])
                # Extract subject from our naming convention: "subjectID_folderName_frame.avi"
                subject_id = clip['frame_dir'].split('_')[0]
                unique_subjects.add(subject_id)

    # 2. Create Integer Label Map
    sorted_labels = sorted(list(unique_labels))
    label_map = {label: idx for idx, label in enumerate(sorted_labels)}
    
    print("\n--- Label to Integer Mapping ---")
    for label, idx in label_map.items():
        print(f"{idx}: {label}")
    print("--------------------------------\n")

    # 3. Perform 8-1-1-1 Cross-Subject Split
    subjects_list = sorted(list(unique_subjects))
    print(f"Found {len(subjects_list)} unique subjects: {subjects_list}")
    
    if len(subjects_list) != 11:
        print(f"[WARNING] Expected 11 subjects, found {len(subjects_list)}.")

    random.seed(seed)
    random.shuffle(subjects_list)

    train_subs = set(subjects_list[0:8])
    valid_subs = set(subjects_list[8:9])
    test_subs  = set(subjects_list[9:10])
    calib_subs = set(subjects_list[10:11])

    # 4. Build the PYSKL Data Structure
    pyskl_data = {
        'split': {
            'train': [],
            'valid': [],
            'test': [],
            'calibration': []
        },
        'annotations': []
    }

    print("\nFormatting arrays to [M, T, V, C] and verifying dimensions...")

    # Counters to verify the split distribution
    split_counts = {'train': 0, 'valid': 0, 'test': 0, 'calibration': 0}

    for clip in all_clips:
        clip_name = clip['frame_dir']
        subject_id = clip_name.split('_')[0]
        
        # Determine which split this clip belongs to
        if subject_id in train_subs:
            target_split = 'train'
        elif subject_id in valid_subs:
            target_split = 'valid'
        elif subject_id in test_subs:
            target_split = 'test'
        elif subject_id in calib_subs:
            target_split = 'calibration'
        else:
            continue 

        # Extract basic info, ensuring strict Python native types
        total_frames = int(clip['total_frames'])
        label_int = int(label_map[clip['label']])
        img_shape = tuple(map(int, clip['img_shape']))
        
        # Build the Tensors. dtype=np.float16 is highly recommended for PYSKL.
        kpts_array = np.array([p['keypoints'] for p in clip['persons']], dtype=np.float16)
        scores_array = np.array([p['keypoint_scores'] for p in clip['persons']], dtype=np.float16)

        # STRICT DIMENSION CHECK: Ensure T matches total_frames
        # kpts_array shape should be [M, T, 17, 2]. 
        # kpts_array.shape[1] is the temporal dimension (T).
        actual_t = kpts_array.shape[1]
        
        if actual_t != total_frames:
            print(f"[ERROR] Dimension mismatch in {clip_name}! Array T={actual_t}, total_frames={total_frames}. Skipping clip.")
            continue

        # Add clip name (identifier) to the appropriate split list
        pyskl_data['split'][target_split].append(clip_name)
        split_counts[target_split] += 1

        # Build the annotation dictionary
        anno = {
            'frame_dir': str(clip_name),
            'label': label_int,
            'total_frames': total_frames,
            'img_shape': img_shape,
            'original_shape': img_shape,
            'keypoint': kpts_array,       # Final shape: [1, T, 17, 2]
            'keypoint_score': scores_array # Final shape: [1, T, 17]
        }
        
        pyskl_data['annotations'].append(anno)

    # 5. Save the final pickle file
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(pyskl_data, f)

    # Save the label mapping as a handy reference text file
    map_path = output_pkl.replace('.pkl', '_label_map.json')
    with open(map_path, 'w') as f:
        json.dump(label_map, f, indent=4)

    print("\n" + "="*40)
    print("PYSKL DATASET PACKAGING COMPLETE")
    print("="*40)
    print(f"Total clips packaged: {len(pyskl_data['annotations'])}")
    print(f"  - Training clips:    {split_counts['train']} (Subjects: {len(train_subs)})")
    print(f"  - Validation clips:  {split_counts['valid']} (Subjects: {len(valid_subs)})")
    print(f"  - Testing clips:     {split_counts['test']} (Subjects: {len(test_subs)})")
    print(f"  - Calibration clips: {split_counts['calibration']} (Subjects: {len(calib_subs)})")
    print("-" * 40)
    print(f"Pickle file saved to: {output_pkl}")
    print(f"Label map saved to:   {map_path}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine YOLO JSONs into a PYSKL cross-subject pickle dataset.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Folder containing cleaned JSONs.")
    parser.add_argument("-o", "--output_pkl", type=str, required=True, help="Path for final .pkl file.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for the 8:1:1:1 split.")
    
    args = parser.parse_args()
    generate_pyskl_pkl(args.input_folder, args.output_pkl, args.seed)