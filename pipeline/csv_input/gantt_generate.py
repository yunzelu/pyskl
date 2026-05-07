import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone

def generate_gantt_chart(json_path, output_image_path):
    print(f"Loading flattened timeline from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("No data found to plot!")
        return

    # FIX 1: Sort the data chronologically based on start time. 
    # This ensures we always find the true first event of the day.
    data = sorted(data, key=lambda x: x.get('start_unix_time', 0))

    # 1. Update the configuration to match our new HAR taxonomy
    config = {
        "Walking":                      {"level": 9,  "color": "#2ecc71"}, # Green
        "Transition-LayFloor-to-Stand": {"level": 8,  "color": "#e67e22"}, # Orange
        "Transition-Sit-to-Stand":      {"level": 7,  "color": "#f39c12"}, # Light Orange
        "Transition-Stand-to-Sit":      {"level": 6,  "color": "#f1c40f"}, # Yellow
        "Transition-LayBed-to-Sit":     {"level": 5,  "color": "#f3ce70"}, # Light Yellow
        "Sit-Stationary":               {"level": 4,  "color": "#1abc9c"}, # Teal
        "Transition-Sit-to-LayBed":     {"level": 3,  "color": "#5dade2"}, # Light Blue
        "Falling":                      {"level": 2,  "color": "#e74c3c"}, # Red (CRITICAL)
        "Multiperson":                  {"level": 1,  "color": "#9b59b6"}, # Purple
        "No detection":                 {"level": 0,  "color": "#bdc3c7"}, # Grey
        "Out-of-Room":                  {"level": 0,  "color": "#bdc3c7"}, # Grey
        # "Lying":                        {"level": 10, "color": "#2980b9"}, # Dark Blue
        # "Lay-Stationary":                 {"level": 10, "color": "#2980b9"}, # Dark Blue
        "LayBed-Stationary":            {"level": 10, "color": "#2980b9"},
        "LayFloor-Stationary":          {"level": 11, "color": "#2980b9"},
    }

    # Define EDT timezone (UTC-4)
    edt_tz = timezone(timedelta(hours=-4))

    fig, ax = plt.subplots(figsize=(15, 6))
    processed_data = []

    # 2. Process each block in the flattened timeline
    for entry in data:
        # label = entry.get('label')
        label = entry.get('action')
        
        if label not in config:
            continue
            
        # Convert Unix timestamps to UTC, then to EDT
        # start_utc = datetime.fromtimestamp(entry['start_time'], tz=timezone.utc)
        # end_utc = datetime.fromtimestamp(entry['end_time'], tz=timezone.utc)
        start_utc = datetime.fromtimestamp(entry['start_unix_time'], tz=timezone.utc)
        end_utc = datetime.fromtimestamp(entry['end_unix_time'], tz=timezone.utc)
        
        start_edt = start_utc.astimezone(edt_tz)
        end_edt = end_utc.astimezone(edt_tz)
        
        # For matplotlib, use naive datetimes (strip tzinfo after conversion)
        start_naive = start_edt.replace(tzinfo=None)
        end_naive = end_edt.replace(tzinfo=None)
        
        start_num = mdates.date2num(start_naive)
        duration = mdates.date2num(end_naive) - start_num

        # Ensure very short actions (like a quick fall) are visible
        duration = max(duration, 30.0 / 86400.0) 
        
        processed_data.append({
            'start': start_naive
        })
        
        # Draw the bar for this action block
        ax.barh(config[label]['level'], duration,
                left=start_num, height=0.6,
                color=config[label]['color'], align='center', linewidth=0)

    if not processed_data:
        print("No valid actions found to plot.")
        return

    # 3. Format the Y-axis
    sorted_config = sorted(config.items(), key=lambda item: item[1]['level'])
    ax.set_yticks([item[1]['level'] for item in sorted_config])
    ax.set_yticklabels([item[0] for item in sorted_config], fontsize=12)

    # 4. DYNAMIC LIMITS: Get absolute midnight to midnight
    # Find the earliest time in our processed data
    min_time = min(p['start'] for p in processed_data)
    
    # Set to 00:00:00 of that day
    midnight_start = min_time.replace(hour=0, minute=0, second=0, microsecond=0)
    # Set to 00:00:00 of the next day
    midnight_end = midnight_start + timedelta(days=1)
    
    # FIX 2: Force the X-axis limits strictly to this 24-hour window
    ax.set_xlim(mdates.date2num(midnight_start), mdates.date2num(midnight_end))
    
    # FIX 3: Remove automatic white space padding at the ends of the X-axis
    ax.margins(x=0)

    # 5. Format the X-axis (Time)
    # Ensure ticks land exactly on the hour, every 3 hours (including midnight edges)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)

    # X-axis label with dynamic date
    base_date = midnight_start.strftime('%Y-%m-%d')
    ax.set_xlabel(f"Time (Date: {base_date} EDT)", fontsize=14, labelpad=15)

    # Final visual touches
    ax.set_title("Activity Timeline - 310 18 - v3 - 11C", fontsize=16, pad=20)
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved successfully to: {output_image_path}")
    # plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Gantt chart from processed JSON.")
    parser.add_argument('--input-json', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--output-image', type=str, required=True, help="Path to save the output chart image")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_gantt_chart(args.input_json, args.output_image)