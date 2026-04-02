import os
import json
import pickle
import numpy as np
import re

# ==========================================
# CONFIGURATION
# ==========================================
DATA_ROOT = 'data/up_dataset'  
OUTPUT_PKL = 'data/har/har_v1.1.pkl'
IMG_SHAPE = (640, 480)  

# 1. CLASS MAPPING: Combine Standing and Walking here
# This allows you to keep your 4 folders, but output 3 classes.
CLASS_MAP = {
    'Lying': 0,
    'Sitting': 1,
    'Standing': 2,
    'Walking': 3  # Mapped to the same integer as Standing!
}

# 2. CROSS-SUBJECT SPLIT: Define which subjects go where
TRAIN_SUBJECTS = ['Subject1', 'Subject2', 'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17', 'Subject8', 'Subject9', 'Subject10', 'Subject11', 'Subject12']
VAL_SUBJECTS   = ['Subject3', 'Subject4']
CALIB_SUBJECTS = ['Subject5']
TEST_SUBJECTS  = ['Subject6', 'Subject7']
# ==========================================

def extract_subject_id(filename):
    """
    Extracts the Subject ID from filenames like:
    'Subject1Activity11Trial1Camera1_mirrored.json' -> Returns 'Subject1'
    'Subject17Activity4Trial3Camera2.json' -> Returns 'Subject17'
    """
    # Searches for the word "Subject" followed by one or more digits (\d+)
    match = re.search(r'(Subject\d+)', filename)
    
    if match:
        return match.group(1) 
    else:
        # Failsafe in case a weird file sneaks into your folder
        raise ValueError(f"Could not find a matching Subject ID in the file: {filename}")

def load_json_sequence(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    T = len(data)
    V = 17
    C = 2
    
    keypoint = np.zeros((1, T, V, C), dtype=np.float32)
    keypoint_score = np.zeros((1, T, V), dtype=np.float32)
    
    for t, frame_data in enumerate(data):
        if frame_data is not None and 'keypoints' in frame_data:
            kpts = np.array(frame_data['keypoints'], dtype=np.float32)
            scores = np.array(frame_data['scores'], dtype=np.float32)
            
            if kpts.shape[0] == V:
                keypoint[0, t, :, :] = kpts
                keypoint_score[0, t, :] = scores
                
    return keypoint, keypoint_score, T

def build_dataset(data_root, img_shape):
    annotations = []
    split = {'train': [], 'val': [], 'calib': [], 'test': []}
    
    # Iterate over the folders we mapped
    for class_folder, mapped_label_idx in CLASS_MAP.items():
        class_dir = os.path.join(data_root, class_folder)
        if not os.path.isdir(class_dir):
            print(f"{class_dir} doesn't exist, skipping.")
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith('.json')]
        
        for file_name in files:
            frame_dir = file_name.replace('.json', '')
            json_path = os.path.join(class_dir, file_name)
            
            # Extract subject ID to figure out which split this file belongs to
            sub_id = extract_subject_id(file_name)
            
            keypoint, keypoint_score, total_frames = load_json_sequence(json_path)
            
            anno = {
                'frame_dir': frame_dir,
                'label': mapped_label_idx, # Using the 3-class mapped integer
                'img_shape': img_shape,
                'original_shape': img_shape,
                'total_frames': total_frames,
                'keypoint': keypoint,
                'keypoint_score': keypoint_score
            }
            annotations.append(anno)
            
            # Route the file to the correct split bucket based on Subject ID
            if sub_id in TRAIN_SUBJECTS:
                split['train'].append(frame_dir)
            elif sub_id in VAL_SUBJECTS:
                split['val'].append(frame_dir)
            elif sub_id in CALIB_SUBJECTS:
                split['calib'].append(frame_dir)
            elif sub_id in TEST_SUBJECTS:
                split['test'].append(frame_dir)
            else:
                print(f"Warning: Subject {sub_id} not found in any split lists. Skipping.")
                
    return {'split': split, 'annotations': annotations}

def save_pkl(dataset_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(dataset_dict, f)
    print(f"Successfully saved to {output_path}")
    print(f"Train samples: {len(dataset_dict['split']['train'])}")
    print(f"Val samples:   {len(dataset_dict['split']['val'])}")
    print(f"Calib samples: {len(dataset_dict['split']['calib'])}")
    print(f"Test samples:  {len(dataset_dict['split']['test'])}")

def main():
    dataset_dict = build_dataset(DATA_ROOT, IMG_SHAPE)
    save_pkl(dataset_dict, OUTPUT_PKL)

if __name__ == '__main__':
    main()