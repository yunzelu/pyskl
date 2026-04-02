import json
import os

def load_data(filepath):
    """Loads JSON data from the specified filepath."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_data(data, filepath):
    """Saves the JSON data to the specified filepath."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n[+] Data successfully saved to {filepath}")

def main():
    # File configuration
    input_file = 'pipeline/csv_input/12-12_result.json'  # Replace with your actual input file name
    output_file = 'pipeline/csv_input/12-12_result_corrected.json' # Saves to a new file to prevent accidental data loss

    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'. Please ensure the file is in the same directory.")
        return

    # Load the initial data
    data = load_data(input_file)
    print(f"Loaded {len(data)} records from {input_file}.")

    while True:
        print("\n" + "-"*30)
        command = input("Press Enter to correct a time range, or type 'q' to quit and save: ").strip().lower()

        if command == 'q':
            save_data(data, output_file)
            break

        # Get the time range
        try:
            start_str = input("Enter the START unix_time: ").strip()
            end_str = input("Enter the END unix_time: ").strip()
            
            start_time = float(start_str)
            end_time = float(end_str)
            
            if start_time > end_time:
                print("Error: Start time must be before or equal to the end time. Let's try again.")
                continue
                
        except ValueError:
            print("Error: Invalid input. Please enter valid numerical values for the unix timestamps.")
            continue

        # Get the new label
        new_label = input("Enter the new action label: ").strip()

        # Iterate and update
        update_count = 0
        for item in data:
            # The prompt requested we check if the start_unix_time falls within the range
            if start_time <= item['start_unix_time'] <= end_time:
                item['action'] = new_label
                update_count += 1

        print(f"--> Success! Updated {update_count} records to '{new_label}'.")

if __name__ == "__main__":
    main()