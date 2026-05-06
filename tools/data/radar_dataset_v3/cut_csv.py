import os
import csv
import argparse
import random

def parse_labels(csv_path):
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            if len(row) >= 2:
                labels.append((int(row[0]), row[1].strip()))
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

def process_videos(root_dir, target_label, folder_filters, add_bg, min_len, max_len):
    random.seed(42)
    target_label_lower = target_label.lower()
    folder_filters = [f.lower() for f in folder_filters]
    output_rows = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_name = os.path.basename(dirpath)
        if any(f in folder_name.lower() for f in folder_filters):
            csv_file = os.path.join(dirpath, 'frame_labels.csv')
            if os.path.exists(csv_file):
                intervals = parse_labels(csv_file)
                for i, interval in enumerate(intervals):
                    if interval['label'].lower() == target_label_lower:
                        t_start, t_end, t_len = interval['start'], interval['end'], interval['length']
                        
                        # --- LOGIC SELECTION ---
                        if add_bg:
                            # Use original logic: add context from i-1 and i+1
                            p_start = intervals[i-1]['start'] if i > 0 and intervals[i-1]['label'].upper() != "DELETE" else t_start
                            n_end = intervals[i+1]['end'] if i < len(intervals)-1 and intervals[i+1]['label'].upper() != "DELETE" else t_end
                            
                            combined_len = n_end - p_start + 1
                            if combined_len < min_len:
                                print(f"Dropped (Short): {folder_name} | {interval['label']} | Len: {combined_len}")
                                continue
                            
                            # UPDATED: Use max_len instead of 100
                            if combined_len >= max_len:
                                # Randomly cut to exactly max_len while keeping target safe
                                s_min = max(p_start, t_end - (max_len - 1))
                                s_max = min(t_start, n_end - (max_len - 1))
                                clip_start = random.randint(s_min, s_max)
                                clip_end = clip_start + (max_len - 1)
                            else:
                                clip_start, clip_end = p_start, n_end
                        else:
                            # Stationary Logic: Only take the labeled period
                            if t_len < min_len:
                                print(f"Dropped (Short): {folder_name} | {interval['label']} | Len: {t_len}")
                                continue
                                
                            # UPDATED: Use max_len instead of 100
                            if t_len >= max_len:
                                # Take the middle max_len frames
                                mid = t_start + (t_len // 2)
                                clip_start = mid - (max_len // 2)
                                clip_end = clip_start + (max_len - 1)
                            else:
                                # Take the whole short period
                                clip_start, clip_end = t_start, t_end

                        output_rows.append({
                            'Folder': folder_name,
                            'VideoFile': 'output_video.avi',
                            'TargetLabel': interval['label'],
                            'ClipStart': clip_start,
                            'ClipEnd': clip_end,
                            'ClipLength': clip_end - clip_start + 1
                        })
    return output_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str, required=True)
    parser.add_argument('-f', '--filters', type=str, nargs='+', required=True)
    parser.add_argument('-d', '--dir', type=str, default='.', help="Root directory containing data.")
    parser.add_argument('--add_bg', type=str, default='False', help="True/False")
    parser.add_argument('--min_len', type=int, default=90)
    # NEW ARGUMENT: Default is now 110 frames
    parser.add_argument('--max_len', type=int, default=110, help="Maximum clip length to extract") 
    parser.add_argument('-o', '--output', type=str, default='output_clips.csv')
    args = parser.parse_args()

    # Convert string to boolean
    should_add_bg = args.add_bg.lower() == 'true'

    results = process_videos(args.dir, args.label, args.filters, should_add_bg, args.min_len, args.max_len)
    
    if results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Done! Saved {len(results)} clips.")

if __name__ == "__main__":
    main()