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
                     f"Time {window['start_unix_time']} - {window['end_unix_time']} | "
                     f"Action: {action_label} ({confidence:.2f})")
        print(log_entry)

        action_results.append({
            "track_id": window['track_id'],
            "start_frame": window['start_frame'],
            "end_frame": window['end_frame'],
            "start_unix_time": window['start_unix_time'],
            "end_unix_time": window['end_unix_time'],
            "action": action_label,
            "confidence": float(confidence)
        })
        
    return action_results

if __name__ == "__main__":
    # 1. Get windows from the bridge (Make sure to point to your CSV file now!)
    windows = create_windows("pipeline/csv_input/12-12_padded.csv", window_size=60, stride=30)
    
    # 2. Run ST-GCN++ 
    action_results = run_action_recognition(
        windows=windows,
        config_path="configs/stgcn++/har4_j.py",
        checkpoint_path="work_dirs/stgcn++/har4_j/epoch_16.pth",
        label_map_path="tools/data/label_map/har4.txt"
    )

    with open("pipeline/csv_input/12-12_result.json", "w") as f:
        json.dump(action_results, f, indent=4)