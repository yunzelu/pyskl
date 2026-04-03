import json
import os
import readline

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_data(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n[+] Data successfully saved to {filepath}")

def input_with_prefill(prompt, prefill):
    """Displays a prompt with pre-filled text that the user can edit."""
    # If there's no prefill data yet, just act like a normal input
    if not prefill:
        return input(prompt)
        
    def hook():
        readline.insert_text(prefill)
        readline.redisplay()
    
    readline.set_pre_input_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_pre_input_hook(None)

def main():
    input_file = 'pipeline/csv_input/12-12_result_v1.1_corrected.json'
    output_file = 'pipeline/csv_input/12-12_result_v1.1_corrected.json'

    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'.")
        return

    data = load_data(input_file)
    print(f"Loaded {len(data)} records.")

    last_end_time_str = ""

    while True:
        print("\n" + "-"*30)
        command = input("Press Enter to correct, or 'q' to save/quit: ").strip().lower()

        if command == 'q':
            save_data(data, output_file)
            break

        try:
            new_label = input("Enter the new action label: ").strip()

            # 1. Pre-fill START with the PREVIOUS END time
            start_str = input_with_prefill("Enter the START unix_time: ", last_end_time_str).strip()

            # 2. Pre-fill END with the CURRENT START time
            # This allows you to just hit Enter if START and END are the same
            end_str = input_with_prefill("Enter the END unix_time:   ", start_str).strip()
            
            start_time = float(start_str)
            end_time = float(end_str)
            
            # Store the current end_str to be used as the next loop's start_str
            last_end_time_str = end_str

            if start_time > end_time:
                print("Error: Start time cannot be after end time.")
                continue
                
        except ValueError:
            print("Error: Invalid numerical values.")
            continue

        update_count = 0
        for item in data:
            if start_time <= item['start_unix_time'] <= end_time:
                item['action'] = new_label
                update_count += 1

        print(f"--> Success! Updated {update_count} records to '{new_label}'.")

if __name__ == "__main__":
    main()