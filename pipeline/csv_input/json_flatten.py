import json
import argparse

def flatten_to_continuous_timeline(input_json, output_json):
    print(f"Loading inference data from: {input_json}")
    with open(input_json, 'r') as f:
        windows = json.load(f)

    if not windows:
        print("No data found in JSON.")
        return

    # 1. Collect all unique time boundaries
    time_boundaries = set()
    for w in windows:
        time_boundaries.add(w['start_unix_time'])
        time_boundaries.add(w['end_unix_time'])
    
    # Sort the boundaries to create a continuous timeline
    sorted_times = sorted(list(time_boundaries))
    
    raw_segments = []

    # 2. Process each tiny time segment
    for i in range(len(sorted_times) - 1):
        start_t = sorted_times[i]
        end_t = sorted_times[i+1]
        mid_t = (start_t + end_t) / 2.0  # Midpoint is highly robust for floating point time

        # Find all windows that are active during this exact midpoint
        active_windows = [w for w in windows if w['start_unix_time'] <= mid_t < w['end_unix_time']]

        # Apply your logic rules
        if len(active_windows) == 0:
            final_label = "No detection"
            
        else:
            # Check how many unique people are in this exact segment
            unique_ids = set(w['track_id'] for w in active_windows)
            
            if len(unique_ids) > 1:
                final_label = "Multiperson"
            else:
                # Same person, but maybe overlapping windows. 
                # Sort by start_unix_time so the latest window is at the end.
                active_windows.sort(key=lambda x: x['start_unix_time'])
                
                # The "last" window overwrites the old one
                final_label = active_windows[-1]['action']
                
        raw_segments.append({
            "start_time": start_t,
            "end_time": end_t,
            "label": final_label
        })

    # 3. Merge consecutive segments that have the exact same label
    merged_timeline = []
    current_segment = raw_segments[0]

    for next_segment in raw_segments[1:]:
        if next_segment['label'] == current_segment['label']:
            # Extend the current segment's end time
            current_segment['end_time'] = next_segment['end_time']
        else:
            # The action changed! Save the current one and start a new one
            merged_timeline.append(current_segment)
            current_segment = next_segment
            
    # Don't forget to add the very last segment
    merged_timeline.append(current_segment)

    # 4. Save to JSON
    with open(output_json, 'w') as f:
        json.dump(merged_timeline, f, indent=4)

    print("\n" + "="*50)
    print("TIMELINE FLATTENING COMPLETE")
    print("="*50)
    print(f"Original overlapping windows: {len(windows)}")
    print(f"Final continuous blocks:      {len(merged_timeline)}")
    print(f"Saved to: {output_json}")
    print("="*50)

def parse_args():
    parser = argparse.ArgumentParser(description="Flatten overlapping JSON action windows into a continuous timeline.")
    parser.add_argument('--input-json', type=str, required=True, help="Path to the input JSON file (from inference)")
    parser.add_argument('--output-json', type=str, required=True, help="Path to save the flattened JSON output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    flatten_to_continuous_timeline(args.input_json, args.output_json)