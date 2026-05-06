import os
import csv
import argparse
import random

def parse_labels(csv_path):
    """Reads the frame_labels.csv and converts it to a list of intervals."""
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Skip header (Frame,Label)
        for row in reader:
            if len(row) >= 2:
                labels.append((int(row[0]), row[1].strip()))
    
    # Sort by frame to ensure correct chronological order
    labels.sort(key=lambda x: x[0])
    
    intervals = []
    for i in range(len(labels) - 1):
        start_frame = labels[i][0]
        end_frame = labels[i+1][0] - 1
        label_name = labels[i][1]
        
        # Stop processing if we hit END
        if label_name.upper() == "END":
            break
            
        intervals.append({
            'label': label_name,
            'start': start_frame,
            'end': end_frame,
            'length': end_frame - start_frame + 1
        })
    return intervals

def process_videos(root_dir, target_label, folder_filters):
    random.seed(42) # Set seed to ensure the same random cuts every time
    target_label = target_label.lower()
    
    # Make all filters lowercase for safe matching
    folder_filters = [f.lower() for f in folder_filters] 
    
    output_rows = []
    
    # Walk through the directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_name = os.path.basename(dirpath)
        
        # NEW LOGIC: Check if the folder matches ANY of the filters provided
        if any(f in folder_name.lower() for f in folder_filters):
            csv_file = os.path.join(dirpath, 'frame_labels.csv')
            
            if os.path.exists(csv_file):
                intervals = parse_labels(csv_file)
                
                for i, interval in enumerate(intervals):
                    if interval['label'].lower() == target_label:
                        target_start = interval['start']
                        target_end = interval['end']
                        target_len = interval['length']
                        
                        clip_start = -1
                        clip_end = -1
                        
                        # CASE 1: Target itself is 100+ frames
                        if target_len >= 100:
                            midpoint = target_start + (target_len // 2)
                            clip_start = midpoint - 50
                            clip_end = clip_start + 99
                            
                        # CASE 2: Target is < 100 frames
                        else:
                            # Get EXACTLY ONE previous activity (i-1)
                            prev_start = target_start
                            if i > 0 and intervals[i-1]['label'].upper() != "DELETE":
                                prev_start = intervals[i-1]['start']
                                
                            # Get EXACTLY ONE next activity (i+1)
                            next_end = target_end
                            if i < len(intervals) - 1 and intervals[i+1]['label'].upper() != "DELETE":
                                next_end = intervals[i+1]['end']
                                
                            combined_len = next_end - prev_start + 1
                            
                            # Verification 1: Print and drop if too short
                            if combined_len < 90:
                                print(f"Dropped: {folder_name} (Frames {target_start}-{target_end}). Combined length is only {combined_len} frames.")
                                continue
                            
                            elif combined_len < 100:
                                # Keep the 90-99 frame clip to zero-pad later
                                clip_start = prev_start
                                clip_end = next_end
                            
                            else:
                                # We have enough frames. Calculate the safe random window.
                                min_possible_start = max(prev_start, target_end - 99)
                                max_possible_start = min(target_start, next_end - 99)
                                
                                clip_start = random.randint(min_possible_start, max_possible_start)
                                clip_end = clip_start + 99
                        
                        rel_target_start = target_start - clip_start
                        rel_target_end = target_end - clip_start
                        
                        output_rows.append({
                            'Folder': folder_name,
                            'VideoFile': 'output_video.avi',
                            'TargetLabel': interval['label'],
                            'ClipStart': clip_start,
                            'ClipEnd': clip_end,
                            'ClipLength': clip_end - clip_start + 1,
                            'Target_Relative_Start': rel_target_start,
                            'Target_Relative_End': rel_target_end
                        })
                        
    return output_rows

def main():
    parser = argparse.ArgumentParser(description="Extract video clips based on CSV labels.")
    parser.add_argument('-d', '--dir', type=str, default='.', help="Root directory containing data.")
    parser.add_argument('-l', '--label', type=str, required=True, help="Target class label to extract.")
    
    # NEW ARGUMENT: nargs='+' allows multiple inputs like: -f "-sit" "-laysofa"
    parser.add_argument('-f', '--filters', type=str, nargs='+', required=True, help="One or more substrings to filter folders.")
    
    parser.add_argument('-o', '--output', type=str, default='output_clips.csv', help="Output CSV filename.")
    
    args = parser.parse_args()
    
    print(f"Scanning for label '{args.label}' in folders containing ANY of: {args.filters}...")
    results = process_videos(args.dir, args.label, args.filters)
    
    if results:
        keys = results[0].keys()
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Success! {len(results)} clips saved to {args.output}")
    else:
        print("No clips found matching your criteria.")

if __name__ == "__main__":
    main()