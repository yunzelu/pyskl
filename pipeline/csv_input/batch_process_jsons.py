import os
import glob
from pathlib import Path

# Import the function from your newly updated script
# Make sure flatten_timeline.py is in the same folder as this script
from json_flatten import flatten_to_continuous_timeline

def process_all_jsons(base_dir):
    print(f"Searching for JSON files in: {base_dir}")
    
    # Use glob to find all .json files in the base directory and all subfolders (recursive)
    search_pattern = os.path.join(base_dir, "**", "*.json")
    all_json_files = glob.glob(search_pattern, recursive=True)
    
    processed_count = 0

    for json_path in all_json_files:
        # Skip files that already have "_f" at the end to prevent double-processing
        if json_path.endswith('_f.json'):
            continue
        if json_path.startswith('gt_'):
            continue

        # Get the path components
        path_obj = Path(json_path)
        folder = path_obj.parent
        file_name_without_ext = path_obj.stem # e.g., "pose_2026-04-18_310_p_r"
        
        # Create the new output file name by adding "_f"
        new_file_name = f"{file_name_without_ext}_f.json"
        output_json_path = folder / new_file_name
        
        print(f"\nProcessing: {json_path}")
        
        # Run the imported flattening function
        try:
            flatten_to_continuous_timeline(str(json_path), str(output_json_path))
            processed_count += 1
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

    print("\n" + "#"*50)
    print(f"BATCH PROCESSING FINISHED.")
    print(f"Successfully processed {processed_count} files.")
    print("#"*50)

if __name__ == "__main__":
    # Define the base directory containing your room subfolders
    BASE_DIR = "/mnt/d/lu/project/auto_labeling_pipeline/data/Willowbend/"
    
    process_all_jsons(BASE_DIR)