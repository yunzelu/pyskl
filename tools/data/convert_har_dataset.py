import os
import json
import pickle
import numpy as np
import random

# ==========================================
# CONFIGURATION
# ==========================================
DATA_ROOT = 'up_dataset'  # Folder containing 'Lying', 'Sitting', etc.
OUTPUT_PKL = 'data/har/har.pkl'
IMG_SHAPE = (640, 480)  # (Height, Width) - Replace with your video resolution

# Define your splits (must sum to <= 1.0)
# Test set will automatically take whatever is leftover.
TRAIN_RATIO = 0.60   # 60% for training
VAL_RATIO = 0.15     # 15% for validation (tuning)
CALIB_RATIO = 0.10   # 10% for conformal calibration
# Remaining 15% will be used for testing

CLASSES = ['Lying', 'Sitting', 'Standing', 'Walking']
# ==========================================

def load_json_sequence(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    T = len(data)
    V = 17
    C = 2
    
    # Initialize with zeros
    # Shape: [M, T, V, C] -> [1, T, 17, 2]
    keypoint = np.zeros((1, T, V, C), dtype=np.float32)
    # Shape: [M, T, V] -> [1, T, 17]
    keypoint_score = np.zeros((1, T, V), dtype=np.float32)
    
    for t, frame_data in enumerate(data):
        if frame_data is not None and 'keypoints' in frame_data:
            kpts = np.array(frame_data['keypoints'], dtype=np.float32)
            scores = np.array(frame_data['scores'], dtype=np.float32)
            
            # Ensure the frame has the right number of joints before assigning
            if kpts.shape[0] == V:
                keypoint[0, t, :, :] = kpts
                keypoint_score[0, t, :] = scores
        else:
            pass
            
    return keypoint, keypoint_score, T

def build_dataset(data_root, classes, img_shape):
    annotations = []
    # NEW: Added calib and test to the split dictionary
    split = {'train': [], 'val': [], 'calib': [], 'test': []}
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            print(class_dir + " doesn't exist")
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith('.json')]
        random.shuffle(files)
        
        total_files = len(files)
        
        # NEW: Calculate cutoff indices for the splits
        train_end = int(total_files * TRAIN_RATIO)
        val_end = train_end + int(total_files * VAL_RATIO)
        calib_end = val_end + int(total_files * CALIB_RATIO)
        
        for i, file_name in enumerate(files):
            frame_dir = file_name.replace('.json', '')
            json_path = os.path.join(class_dir, file_name)
            
            keypoint, keypoint_score, total_frames = load_json_sequence(json_path)
            
            anno = {
                'frame_dir': frame_dir,
                'label': label_idx,
                'img_shape': img_shape,
                'original_shape': img_shape,
                'total_frames': total_frames,
                'keypoint': keypoint,
                'keypoint_score': keypoint_score
            }
            annotations.append(anno)
            
            # NEW: Distribute files into the 4 buckets based on the calculated cutoffs
            if i < train_end:
                split['train'].append(frame_dir)
            elif i < val_end:
                split['val'].append(frame_dir)
            elif i < calib_end:
                split['calib'].append(frame_dir)
            else:
                split['test'].append(frame_dir)
                
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
    # Pass the updated ratios implicitly via the global variables
    dataset_dict = build_dataset(DATA_ROOT, CLASSES, IMG_SHAPE)
    save_pkl(dataset_dict, OUTPUT_PKL)

if __name__ == '__main__':
    main()