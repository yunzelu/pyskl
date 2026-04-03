import json

def extract_continuous_intervals(json_path, output_json_path=None):
    # Load the raw prediction data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Dictionary to group raw start/end times by action
    raw_intervals = {}

    for item in data:
        # 1. Skip track_id 0
        if str(item["track_id"]) == "-1":
            continue

        action = item["action"]
        
        # Group Standing and Walking together
        if action in ["Standing", "Walking", "Standing/Walking"]:
            action = "Standing/Walking"

        start_t = item["start_unix_time"]
        end_t = item["end_unix_time"]

        if action not in raw_intervals:
            raw_intervals[action] = []
        
        raw_intervals[action].append([start_t, end_t])

    clean_blocks = []

    # 2. Merge overlapping intervals for each action
    for act, intervals in raw_intervals.items():
        # Sort intervals chronologically by their start time
        intervals.sort(key=lambda x: x[0])
        
        if not intervals:
            continue

        # Start the first continuous block
        current_start, current_end = intervals[0]

        for next_start, next_end in intervals[1:]:
            # If the next block overlaps with or touches the current block
            # (e.g., the 0.5s overlap, or two people doing it at the same time)
            if next_start <= current_end:
                # Extend the current block's end time to encompass both
                current_end = max(current_end, next_end)
            else:
                # No overlap means the continuous action broke. 
                # Save the completed block and start tracking a new one.
                clean_blocks.append({
                    "action": act,
                    "start_unix_time": current_start,
                    "end_unix_time": current_end,
                    "duration_seconds": round(current_end - current_start, 4)
                })
                current_start, current_end = next_start, next_end
        
        # Don't forget to append the very last block for this action
        clean_blocks.append({
            "action": act,
            "start_unix_time": current_start,
            "end_unix_time": current_end,
            "duration_seconds": round(current_end - current_start, 4)
        })

    # 3. Sort all the final clean blocks chronologically so the timeline makes sense
    clean_blocks.sort(key=lambda x: x["start_unix_time"])

    # 4. Save to a new JSON file if an output path was provided
    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(clean_blocks, f, indent=4)
        print(f"Saved {len(clean_blocks)} clean continuous blocks to {output_json_path}")

    return clean_blocks

# --- Run the script ---
if __name__ == "__main__":
    input_file = 'pipeline/csv_input/12-12_result_v1.1_corrected.json'
    output_file = 'pipeline/csv_input/12-12_annotation.json'
    
    result_blocks = extract_continuous_intervals(input_file, output_file)
    
    # Print a quick preview of the timeline
    print("\n--- Clean Event Timeline ---")
    for block in result_blocks:
        print(f"[{block['start_unix_time']:.2f} -> {block['end_unix_time']:.2f}] "
              f"{block['action']} (Duration: {block['duration_seconds']:.2f}s)")