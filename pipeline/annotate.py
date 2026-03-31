import json
import os

def build_ground_truth(output_path):
    # Define your 4 custom HAR classes here for rapid data entry
    action_map = {
        "1": "Walking",
        "2": "Standing",
        "3": "Sitting",
        "4": "Lying"
    }
    
    # Load existing data so you can stop and resume labeling later
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            gt_data = json.load(f)
        print(f"\n[SUCCESS] Loaded {len(gt_data)} existing segments from {output_path}.")
    else:
        gt_data = []
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"\n[INFO] Starting fresh. Will create new file at {output_path}.")
        
    print("\n--- Quick Labeler Started ---")
    print("Classes: " + ", ".join([f"[{k}] {v}" for k, v in action_map.items()]))
    print("Type 'q' at any prompt to save and quit.\n")

    while True:
        try:
            track_id = input("Track ID: ").strip()
            if track_id.lower() == 'q': break
            
            act_num = input("Action (1-4): ").strip()
            if act_num.lower() == 'q': break
            action_name = action_map.get(act_num, None)
            
            start_f = input("Start Frame: ").strip()
            if start_f.lower() == 'q': break
            
            end_f = input("End Frame: ").strip()
            if end_f.lower() == 'q': break
            
            # Build the block
            segment = {
                "track_id": int(track_id),
                "action": action_name,
                "start_frame": int(start_f),
                "end_frame": int(end_f)
            }
            
            gt_data.append(segment)
            print(f"Saved: Person {track_id} | {action_name} | {start_f} -> {end_f}\n")
            
        except ValueError:
            print("Invalid input, please enter numbers only. Try again.\n")

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(gt_data, f, indent=2)
    print(f"\nGround truth successfully saved to {output_path}")

if __name__ == "__main__":
    default_path = "pipeline/gt_WIN_20260326_15_06_23_Pro.json"
    
    print("=== Ground Truth Annotator ===")
    user_path = input(f"Enter the path to the ground truth JSON file\n(Press Enter to use default: '{default_path}'): ").strip()
    
    # Use default if the user just presses Enter
    final_path = user_path if user_path else default_path
    
    build_ground_truth(final_path)