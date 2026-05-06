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

    # 1. Update the configuration to match our new HAR taxonomy
    # Levels dictate the vertical position (higher number = higher on the chart)
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
        # "No detection":                 {"level": 0,  "color": "#bdc3c7"}, # Grey
        "Out-of-Room":                  {"level": 0,  "color": "#bdc3c7"}, # Grey
        # "Lying":                        {"level": 10, "color": "#2980b9"}, # Dark Blue
        "LayBed-Stationary":            {"level": 10, "color": "#2980b9"},
        "LayFloor-Stationary":          {"level": 11, "color": "#2980b9"},
    }

    # Define EDT timezone (UTC-4)
    edt_tz = timezone(timedelta(hours=-4))

    fig, ax = plt.subplots(figsize=(15, 6))
    processed_data = []

    # 2. Process each block in the flattened timeline
    for entry in data:
        # Use the new keys from our flatten script
        # label = entry['label']
        label = entry['action']
        
        # Fallback just in case an unknown label appears
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

        # Ensure very short actions (like a quick fall) are visible (minimum 10 seconds)
        duration = max(duration, 30.0 / 86400.0) 
        
        processed_data.append({
            'start': start_naive
        })
        
        # Draw the bar for this action block
        ax.barh(config[label]['level'], duration,
                left=start_num, height=0.6,
                color=config[label]['color'], align='center', linewidth=0)

    # 3. Format the Y-axis
    # Sort the config so the labels match the level heights correctly
    sorted_config = sorted(config.items(), key=lambda item: item[1]['level'])
    ax.set_yticks([item[1]['level'] for item in sorted_config])
    ax.set_yticklabels([item[0] for item in sorted_config], fontsize=12)

    # 4. Format the X-axis (Time)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)

    # 5. DYNAMIC LIMITS: Automatically set the view from midnight to midnight of the active day
    first_event_time = processed_data[0]['start']
    midnight_start = first_event_time.replace(hour=0, minute=0, second=0, microsecond=0)
    midnight_end = midnight_start + timedelta(days=1)
    
    ax.set_xlim(mdates.date2num(midnight_start), mdates.date2num(midnight_end))

    # X-axis label with dynamic date
    base_date = midnight_start.strftime('%Y-%m-%d')
    ax.set_xlabel(f"Time (Date: {base_date} EDT)", fontsize=14, labelpad=15)

    # Final visual touches
    ax.set_title("Activity Timeline - 310 18 GT - 12C", fontsize=16, pad=20)
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