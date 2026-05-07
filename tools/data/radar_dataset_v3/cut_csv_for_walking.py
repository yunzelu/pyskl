import os
import csv
import argparse

def parse_labels(csv_path):
    """Reads the frame_labels.csv and converts it to a list of intervals."""
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            if len(row) >= 2:
                labels.append((int(row[0]), row[1].strip()))
    
    # Sort by frame just to be safe
    labels.sort(key=lambda x: x[0])
    
    intervals = []
    for i in range(len(labels) - 1):
        start_frame = labels[i][0]
        end_frame = labels[i+1][0] - 1
        label_name = labels[i][1]
        
        if label_name.upper() == "END":
            break
            
        intervals.append({
            'label': label_name,
            'start': start_frame,
            'end': end_frame,
            'length': end_frame - start_frame + 1
        })
    return intervals

def process_long_videos(root_dir, target_label, folder_filters, min_len, max_len, stride):
    target_label_lower = target_label.lower()
    folder_filters = [f.lower() for f in folder_filters]
    output_rows = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_name = os.path.basename(dirpath)
        
        if any(f in folder_name.lower() for f in folder_filters):
            csv_file = os.path.join(dirpath, 'frame_labels.csv')
            
            if os.path.exists(csv_file):
                intervals = parse_labels(csv_file)
                
                for interval in intervals:
                    if interval['label'].lower() == target_label_lower:
                        t_start = interval['start']
                        t_end = interval['end']
                        
                        current_start = t_start
                        
                        # SLIDING WINDOW LOGIC
                        # Keep sliding the window forward as long as the remaining frames 
                        # are at least equal to your minimum length requirement (min_len)
                        while (t_end - current_start + 1) >= min_len:
                            
                            clip_end = current_start + (max_len - 1)
                            
                            # If the clip goes slightly past the end of the labeled action, 
                            # shrink it to exactly match the end frame.
                            if clip_end > t_end:
                                clip_end = t_end
                                
                            output_rows.append({
                                'Folder': folder_name,
                                'VideoFile': 'output_video.avi',
                                'TargetLabel': interval['label'],
                                'ClipStart': current_start,
                                'ClipEnd': clip_end,
                                'ClipLength': clip_end - current_start + 1
                            })
                            
                            # Move the window forward by the stride amount
                            current_start += stride
                            
    return output_rows

def main():
    parser = argparse.ArgumentParser(description="Cut long activities into fixed-size windows.")
    parser.add_argument('-l', '--label', type=str, required=True, help="Target class label to extract.")
    parser.add_argument('-f', '--filters', type=str, nargs='+', required=True, help="Substring to filter folders.")
    parser.add_argument('--min_len', type=int, default=90, help="Minimum valid frames required to keep a clip.")
    parser.add_argument('--max_len', type=int, default=100, help="Maximum clip length to extract.") 
    parser.add_argument('-d', '--dir', type=str, default='.', help="Root directory containing data.")
    # NEW ARGUMENT: Stride controls the overlap of your windows
    parser.add_argument('-s', '--stride', type=int, default=100, help="Step size for the sliding window.")
    
    parser.add_argument('-o', '--output', type=str, default='output_clips.csv', help="Output CSV filename.")
    
    args = parser.parse_args()

    results = process_long_videos(args.dir, args.label, args.filters, args.min_len, args.max_len, args.stride)
    
    if results:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Success! Sliced the long data into {len(results)} clips.")
    else:
        print("No clips found matching your criteria.")

if __name__ == "__main__":
    main()