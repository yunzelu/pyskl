import json
import pickle
import numpy as np
import os
import glob
import argparse
from collections import defaultdict

LABEL_MAP = {
    "Walking": 0,
    "Falling": 1,
    "Transition-LayFloor-to-Stand": 2,
    "Transition-Stand-to-Sit": 3,
    "Transition-Sit-to-LayBed": 4, 
    "Transition-LayBed-to-Sit": 5,
    "Transition-LayFloor-to-Stand": 6,
    "Sit-Stationary": 7,
    "Lay-Stationary": 8
}

def convert_folder_to_pkl(json_folder, output_dir, img_size=(480, 640)):
    print(f"Scanning folder {json_folder} for JSON files...")
    
    # Find all .json files in the folder
    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    
    if not json_files:
        print("Error: No JSON files found in this folder!")
        return

    annotations = []
    subject_to_clips = defaultdict(list)
    
    # Process each file one by one
    for json_path in json_files:
        # Get the file name without the ".json" part
        base_name = os.path.basename(json_path)
        class_name_string = base_name.replace('.json', '')
        
        # Translate the string into an integer
        label_int = LABEL_MAP.get(class_name_string, -1)
        
        if label_int == -1:
            print(f"WARNING: File '{base_name}' does not match any class in LABEL_MAP. Skipping this file.")
            continue
            
        print(f"Reading {base_name} -> Assigned Label Number: {label_int}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        for clip in data:
            # Create a unique ID for every clip
            clip_id = f"{clip['folder']}_{clip['clip_start']}_{clip['clip_end']}"
            subject = clip['subject_id']
            
            # Save which subject did which clip
            subject_to_clips[subject].append(clip_id)
            
            T = clip['saved_length']
            
            # Initialize NumPy arrays with zeros
            keypoint = np.zeros((1, T, 17, 2), dtype=np.float32)
            keypoint_score = np.zeros((1, T, 17), dtype=np.float32)
            
            # Fill the arrays
            for t, frame in enumerate(clip['frames']):
                kpts = frame['keypoints']
                scores = frame['keypoint_scores']
                
                if len(kpts) == 17:
                    for v in range(17):
                        keypoint[0, t, v, 0] = kpts[v][0] 
                        keypoint[0, t, v, 1] = kpts[v][1] 
                        keypoint_score[0, t, v] = scores[v]   
                        
            # Create the PySKL annotation dictionary
            ann = {
                'frame_dir': clip_id,
                'label': label_int, # Use the number we found from the filename
                'img_shape': img_size,
                'original_shape': img_size,
                'total_frames': T,
                'keypoint': keypoint,
                'keypoint_score': keypoint_score
            }
            annotations.append(ann)

    # 2. Build the 11 LOSO-CV Splits
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subjects = list(subject_to_clips.keys())
    print(f"\nFound {len(subjects)} unique subjects: {subjects}")
    
    for test_subject in subjects:
        train_list = []
        val_list = subject_to_clips[test_subject] 
        
        # 10 subjects for training
        for sub, clips in subject_to_clips.items():
            if sub != test_subject:
                train_list.extend(clips)
                
        # Create PySKL split dictionary
        split_dict = {
            'train': train_list,
            'val': val_list  
        }
        
        pkl_data = {
            'split': split_dict,
            'annotations': annotations 
        }
        
        # Save to file
        out_file = os.path.join(output_dir, f'invisiguard_loso_{test_subject}.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump(pkl_data, f)
            
        print(f"Saved {out_file} (Train: {len(train_list)} clips | Test: {len(val_list)} clips)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CHANGED: We now ask for a directory, not a single file
    parser.add_argument('-d', '--dir', type=str, required=True, help="Folder containing your class JSON files.")
    parser.add_argument('-o', '--outdir', type=str, default='pyskl_data', help="Folder to save PKLs.")
    parser.add_argument('--width', type=int, default=640, help="Original video width.")
    parser.add_argument('--height', type=int, default=480, help="Original video height.")
    args = parser.parse_args()
    
    convert_folder_to_pkl(args.dir, args.outdir, img_size=(args.height, args.width))