import json
from pathlib import Path

def batch_remap_labels(root_folder):
    # 1. Define the mapping dictionary based on your 9 classes to 4 classes
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

    # 2. Set the starting folder path
    root_path = Path(root_folder)
    
    # 3. Find all files ending in '_f.json' in this folder and all subfolders
    # rglob means "recursive search"
    target_files = list(root_path.rglob("*_r3.1_f.json"))

    if not target_files:
        print(f"No files ending with '_f.json' found in {root_folder}")
        return

    print(f"Found {len(target_files)} files. Starting to process...\n")

    # 4. Loop through every file we found
    for file_path in target_files:
        # Open and load the original JSON data
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Create a new list for the mapped data
        new_data = []
        for entry in data:
            original_action = entry.get("label")
            
            # Map to the new class. If the action is not in the dictionary, keep the old name.
            new_action = mapping.get(original_action, original_action)
            
            # Copy the entry so we don't destroy the original data, then update the action
            new_entry = entry.copy()
            new_entry["label"] = new_action
            new_data.append(new_entry)

        # 5. Create the new file name
        # file_path.name gets just the file name (e.g., "video1_f.json")
        # We replace "_f.json" with "_4c.json"
        new_filename = file_path.name.replace('.json', '_5c.json')
        
        # file_path.parent gets the folder where the original file is located
        new_file_path = file_path.parent / new_filename

        # 6. Save the new JSON file
        with open(new_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4)
        
        print(f"Saved: {new_file_path}")

    print("\nSuccess! All files have been processed.")

# ==========================================
# How to use the script:
# Change the path below to the folder containing your JSON files.
# Windows example: r"C:\Users\Name\Documents\InvisiGuard_Data"
# Mac/Linux example: "/home/name/InvisiGuard_Data"
# ==========================================

folder_to_search = "/mnt/d/lu/project/auto_labeling_pipeline/data/Willowbend/310/" # <-- Put your folder path here
batch_remap_labels(folder_to_search)