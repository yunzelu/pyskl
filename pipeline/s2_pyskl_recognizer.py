import mmcv
import torch
from pyskl.apis import init_recognizer, inference_recognizer
from formatter import create_windows
import json

def run_action_recognition(windows, config_path, checkpoint_path, label_map_path):
    # Initialize the ST-GCN++ model
    config = mmcv.Config.fromfile(config_path)
    
    # Remove DecompressPose if it exists in the test pipeline, as we provide raw arrays
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_recognizer(config, checkpoint_path, device)
    
    # Load your custom HAR dataset labels
    label_map = [x.strip() for x in open(label_map_path).readlines()]
    
    # results_log = []
    action_results = []
    
    for window in windows:
        fake_anno = window["fake_anno"]
        
        # inference_recognizer accepts the fake_anno dictionary directly
        result = inference_recognizer(model, fake_anno)
        
        # result is a list of tuples: (class_index, score)
        top_prediction = result[0] 
        action_label = label_map[top_prediction[0]]
        confidence = top_prediction[1]
        
        log_entry = (f"Person {window['track_id']} | "
                     f"Frames {window['start_frame']}-{window['end_frame']} | "
                     f"Action: {action_label} ({confidence:.2f})")
        print(log_entry)
        # results_log.append(log_entry)

        action_results.append({
            "track_id": window['track_id'],
            "start_frame": window['start_frame'],
            "end_frame": window['end_frame'],
            "action": action_label,
            "confidence": float(confidence)
        })
        
    return action_results

if __name__ == "__main__":
    # 1. Get windows from the bridge
    windows = create_windows("pipeline/WIN_20260326_15_06_23_Pro.json", window_size=60, stride=30)
    
    # 2. Run ST-GCN++ 
    action_results = run_action_recognition(
        windows=windows,
        config_path="configs/stgcn++/har4_j.py",
        checkpoint_path="work_dirs//stgcn++/har4_j/latest.pth",
        label_map_path="tools/data/label_map/har4.txt"
    )

    with open("pipeline/s2_WIN_20260326_15_06_23_Pro.json", "w") as f:
        json.dump(action_results, f)