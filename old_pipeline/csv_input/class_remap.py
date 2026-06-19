import json
import os

def remap_labels(input_file):
    # Define the mapping dictionary
    mapping = {
        "Walking": "Standing / Walking",
        "Transition-Stand-to-Sit": "Standing / Walking",
        "Transition-Sit-to-Stand": "Standing / Walking",
        "Sit-Stationary": "Sitting",
        "Lay-Stationary": "Lying",
        "Transition-Sit-to-LayBed": "Lying",
        "Transition-LayBed-to-Sit": "Lying",
        "Transition-LayFloor-to-Stand": "Lying",
        "Falling": "Fall",
        "LayBed-Stationary": "Lying",
        "LayFloor-Stationary": "Lying",
    }

    # Load the original JSON data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    # Process and remap
    new_data = []
    for entry in data:
        original_action = entry.get("action")
        # Map the action if it exists in our dictionary; otherwise, keep original
        new_action = mapping.get(original_action, original_action)
        
        new_entry = entry.copy()
        new_entry["action"] = new_action
        new_data.append(new_entry)

    # Generate new filename
    file_name, file_ext = os.path.splitext(input_file)
    output_file = f"{file_name}_5c{file_ext}"

    # Save the new JSON
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    print(f"Success! New labels saved to: {output_file}")

# Example usage:
remap_labels('/mnt/d/lu/project/auto_labeling_pipeline/data/Willowbend/310/gt_pose_2026-04-18_310.json')