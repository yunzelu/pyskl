import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run PySKL ST-GCN++ Inference for InvisiGuard")
    
    # Required arguments
    parser.add_argument('--csv-path', type=str, required=True, help="Path to the input CSV file containing YOLO pose data")
    parser.add_argument('--out-json', type=str, required=True, help="Path where the output JSON will be saved")
    parser.add_argument('--config', type=str, required=True, help="Path to the PySKL config (.py) file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model weights (.pth)")
    parser.add_argument('--label-map', type=str, required=True, help="Path to the label map text file")
    
    # Optional arguments with default values
    parser.add_argument('--window-size', type=int, default=100, help="Clip length for the model (default: 100)")
    parser.add_argument('--stride', type=int, default=30, help="Stride step for sliding window (default: 30)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    print(f"Loading data from {args.csv_path}...")
    print(f"Using Window Size: {args.window_size}, Stride: {args.stride}")

    # 1. Get windows from the bridge using the parsed arguments
    windows = create_windows(
        csv_path=args.csv_path, 
        window_size=args.window_size, 
        stride=args.stride
    )
    
    # 2. Run ST-GCN++ 
    print("Starting ST-GCN++ inference...")
    action_results = run_action_recognition(
        windows=windows,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        label_map_path=args.label_map
    )

    # 3. Save the results to the specified output path
    with open(args.out_json, "w") as f:
        json.dump(action_results, f, indent=4)
        
    print(f"Inference complete! Results successfully saved to: {args.out_json}")