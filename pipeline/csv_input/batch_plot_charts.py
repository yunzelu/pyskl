import os
import glob
from pathlib import Path

# Import the plotting function from your updated script
from gantt_generate import generate_gantt_chart

def process_all_charts(base_dir):
    print(f"Searching for processed JSON files (*_f.json) in: {base_dir}")
    
    # Use glob to find all files ending in _f.json in all subfolders
    search_pattern = os.path.join(base_dir, "**", "*_f.json")
    all_json_files = glob.glob(search_pattern, recursive=True)
    
    processed_count = 0

    for json_path in all_json_files:
        path_obj = Path(json_path)
        folder = path_obj.parent
        file_name = path_obj.name
        
        # Replace "_f.json" with "_g.png" to create the new name
        new_file_name = file_name.replace('_f.json', '_g.png')
        output_image_path = folder / new_file_name
        
        print(f"\nGenerating chart for: {json_path}")
        print(f"Output will be saved as: {output_image_path}")
        
        # Run the imported plotting function
        try:
            generate_gantt_chart(str(json_path), str(output_image_path))
            processed_count += 1
            # Clear the current plot figure from memory so they don't overlap in the loop
            import matplotlib.pyplot as plt
            plt.close('all') 
        except Exception as e:
            print(f"Error generating chart for {json_path}: {e}")

    print("\n" + "#"*50)
    print("BATCH CHART GENERATION FINISHED.")
    print(f"Successfully created {processed_count} charts.")
    print("#"*50)

if __name__ == "__main__":
    # Define the base directory containing your room subfolders
    BASE_DIR = "/mnt/d/lu/project/auto_labeling_pipeline/data/Willowbend/"
    
    process_all_charts(BASE_DIR)