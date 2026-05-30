#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 41)]
ALL_ACTIONS = [f"A{i:02d}" for i in range(1, 28)]

DAILY_ACTIONS = [
    "A02", "A03", "A04", "A05",
    "A13", "A14", "A17", "A18", "A19",
    "A20", "A21", "A22", "A23", "A27"
]

REHAB_ACTIONS = [
    "A01", "A06", "A07", "A08", "A09", "A10",
    "A11", "A12", "A15", "A16", "A24", "A25", "A26"
]

# Same cross-subject split shown in the MMFi config.yaml
CROSS_SUBJECT_TRAIN = [
    "S01", "S02", "S03", "S04",
    "S06", "S07", "S08", "S09",
    "S11", "S12", "S13", "S14",
    "S16", "S17", "S18", "S19",
    "S21", "S22", "S23", "S24",
    "S26", "S27", "S28", "S29",
    "S31", "S32", "S33", "S34",
    "S36", "S37", "S38", "S39"
]

CROSS_SUBJECT_VAL = [
    "S05", "S10", "S15", "S20",
    "S25", "S30", "S35", "S40"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MMFi RGB 2D keypoints to PYSKL pkl format."
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to MMFi dataset root, e.g. /path/MMFi_Dataset"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output pkl path, e.g. data/mmfi/mmfi_rgb_pyskl_xsub.pkl"
    )

    parser.add_argument(
        "--protocol",
        default="protocol3",
        choices=["protocol1", "protocol2", "protocol3"],
        help="protocol1=daily, protocol2=rehab, protocol3=all actions"
    )

    parser.add_argument(
        "--split",
        default="cross_subject",
        choices=["cross_subject", "cross_scene"],
        help="MMFi split strategy"
    )

    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "segment"],
        help="full: use full E/S/A sequence; segment: crop by CSV"
    )

    parser.add_argument(
        "--segment-csv",
        default=None,
        help="CSV containing action segment start/end frames. Required if mode=segment."
    )

    parser.add_argument(
        "--index-base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Frame index base used in segment CSV. Use 1 if CSV uses frame001-style numbering."
    )

    parser.add_argument(
        "--end-inclusive",
        action="store_true",
        help="Use this if the end frame in CSV is inclusive."
    )

    parser.add_argument(
        "--img-height",
        type=int,
        default=None,
        help="Original image height. If omitted, the script tries to infer a rough value."
    )

    parser.add_argument(
        "--img-width",
        type=int,
        default=None,
        help="Original image width. If omitted, the script tries to infer a rough value."
    )

    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Drop samples shorter than this number of frames."
    )

    parser.add_argument(
        "--save-label-map",
        default=None,
        help="Optional path to save action-to-label mapping as JSON."
    )

    return parser.parse_args()


def get_actions(protocol):
    if protocol == "protocol1":
        return DAILY_ACTIONS
    if protocol == "protocol2":
        return REHAB_ACTIONS
    return ALL_ACTIONS


def scene_of_subject(subject):
    sid = int(subject[1:])
    if 1 <= sid <= 10:
        return "E01"
    if 11 <= sid <= 20:
        return "E02"
    if 21 <= sid <= 30:
        return "E03"
    if 31 <= sid <= 40:
        return "E04"
    raise ValueError(f"Unknown subject: {subject}")


def normalize_token(value, prefix, width=2):
    """
    Convert values such as '1', '01', 'A1', 'A01' to 'A01'.
    """
    value = str(value).strip()

    if value.upper().startswith(prefix):
        num_part = value[1:]
    else:
        num_part = value

    num = int(re.findall(r"\d+", num_part)[0])
    return f"{prefix}{num:0{width}d}"


def frame_number(path):
    """
    Extract frame number from names like frame001.npy.
    """
    m = re.search(r"(\d+)", path.stem)
    if m is None:
        return 0
    return int(m.group(1))


def load_rgb_keypoint_sequence(rgb_dir):
    """
    Load MMFi rgb keypoint sequence.

    Expected per-frame shape:
        [17, 2]

    Output:
        keypoint: [T, 17, 2]
    """
    rgb_dir = Path(rgb_dir)
    frame_files = sorted(rgb_dir.glob("frame*.npy"), key=frame_number)

    if len(frame_files) == 0:
        raise FileNotFoundError(f"No frame*.npy found in {rgb_dir}")

    frames = []

    for f in frame_files:
        if f.stat().st_size == 0:
            continue

        arr = np.load(f)

        # Common expected shape: [17, 2]
        if arr.shape == (17, 2):
            kp = arr

        # Sometimes there may be an extra person dimension: [1, 17, 2]
        elif arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 17 and arr.shape[2] >= 2:
            kp = arr[0, :, :2]

        # Sometimes it may contain score: [17, 3]
        elif arr.ndim == 2 and arr.shape[0] == 17 and arr.shape[1] >= 2:
            kp = arr[:, :2]

        else:
            raise ValueError(f"Unexpected keypoint shape {arr.shape} in {f}")

        frames.append(kp.astype(np.float32))

    if len(frames) == 0:
        raise ValueError(f"All frame files are empty in {rgb_dir}")

    return np.stack(frames, axis=0)  # [T, 17, 2]


def infer_img_shape(kp_seq, img_height=None, img_width=None):
    """
    PYSKL PreNormalize2D needs img_shape.

    If you know the real image size, pass it from command line.
    If coordinates are normalized to [0, 1], this function returns (1, 1).
    Otherwise, it roughly infers size from max coordinates.
    """
    if img_height is not None and img_width is not None:
        return (img_height, img_width)

    max_x = float(np.nanmax(kp_seq[..., 0]))
    max_y = float(np.nanmax(kp_seq[..., 1]))

    # Likely normalized coordinates
    if max_x <= 2.0 and max_y <= 2.0:
        return (1, 1)

    # Fallback: rough per-sequence estimate
    # Better: pass real image height/width explicitly.
    h = int(np.ceil(max_y + 1))
    w = int(np.ceil(max_x + 1))

    h = max(h, 1)
    w = max(w, 1)
    return (h, w)


def infer_column(fieldnames, candidates, required=True):
    lower_map = {name.lower().strip(): name for name in fieldnames}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise KeyError(
            f"Cannot infer CSV column. Candidates={candidates}, "
            f"available={fieldnames}"
        )

    return None


def parse_segments_string(segment_str):
    """
    Parse a segment string like:
        "1-7; 8-15; 16-21"

    Return:
        [(1, 7), (8, 15), (16, 21)]
    """
    segments = []

    if segment_str is None:
        return segments

    segment_str = segment_str.strip()

    if segment_str == "":
        return segments

    parts = segment_str.split(";")

    for part in parts:
        part = part.strip()

        if part == "":
            continue

        if "-" not in part:
            raise ValueError(f"Invalid segment format: {part}")

        start_str, end_str = part.split("-", 1)

        start = int(start_str.strip())
        end = int(end_str.strip())

        if end < start:
            raise ValueError(f"Invalid segment range: {part}")

        segments.append((start, end))

    return segments


def load_segments_csv(csv_path):
    """
    Load MMFi segmentation CSV.

    Expected format:

        Environment,Student,Action,Segments
        E01,S01,A01,1-7; 8-15; 16-21
        E01,S01,A02,1-24; 25-50

    Return:
        segment_dict[(scene, subject, action)] = [(start, end), ...]
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    segment_dict = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        env_col = infer_column(
            fieldnames,
            ["Environment", "scene", "environment", "env", "E", "scene_id"]
        )

        subject_col = infer_column(
            fieldnames,
            ["Student", "subject", "sub", "subject_id", "S"]
        )

        action_col = infer_column(
            fieldnames,
            ["Action", "action", "activity", "action_id", "A", "label"]
        )

        segments_col = infer_column(
            fieldnames,
            ["Segments", "segments", "segment", "Segment"]
        )

        for row in reader:
            scene = normalize_token(row[env_col], "E", width=2)
            subject = normalize_token(row[subject_col], "S", width=2)
            action = normalize_token(row[action_col], "A", width=2)

            segments = parse_segments_string(row[segments_col])

            if len(segments) == 0:
                continue

            segment_dict[(scene, subject, action)].extend(segments)

    return segment_dict


def get_split_name(scene, subject, split):
    if split == "cross_subject":
        if subject in CROSS_SUBJECT_TRAIN:
            return "train"
        if subject in CROSS_SUBJECT_VAL:
            return "val"
        return None

    if split == "cross_scene":
        if scene in ["E01", "E02", "E03"]:
            return "train"
        if scene == "E04":
            return "val"
        return None

    raise ValueError(split)


def make_annotation(frame_dir, label, kp_seq, img_shape):
    """
    kp_seq: [T, 17, 2]

    PYSKL expects:
        keypoint: [M, T, V, C]
    """
    assert kp_seq.ndim == 3
    assert kp_seq.shape[1:] == (17, 2)

    keypoint = kp_seq[None, ...].astype(np.float32)  # [1, T, 17, 2]

    return {
        "frame_dir": frame_dir,
        "total_frames": int(kp_seq.shape[0]),
        "img_shape": tuple(img_shape),
        "original_shape": tuple(img_shape),
        "label": int(label),
        "keypoint": keypoint,
    }


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    actions = get_actions(args.protocol)
    action_to_label = {action: i for i, action in enumerate(actions)}

    if args.save_label_map is not None:
        label_map_path = Path(args.save_label_map)
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_map_path, "w") as f:
            json.dump(action_to_label, f, indent=2)

    if args.mode == "segment":
        if args.segment_csv is None:
            raise ValueError("--segment-csv is required when --mode segment")
        segment_dict = load_segments_csv(args.segment_csv)
    else:
        segment_dict = None

    split_dict = {
        "train": [],
        "val": [],
        # PYSKL can use val as test if you do not have a separate test split.
        "test": []
    }

    annotations = []

    length_counter = Counter()
    split_counter = Counter()
    label_counter = Counter()
    missing_rgb_dirs = []
    missing_segments = []

    for scene_dir in sorted(data_root.glob("E*")):
        if not scene_dir.is_dir():
            continue

        scene = scene_dir.name

        for subject_dir in sorted(scene_dir.glob("S*")):
            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name

            for action_dir in sorted(subject_dir.glob("A*")):
                if not action_dir.is_dir():
                    continue

                action = action_dir.name

                if action not in action_to_label:
                    continue

                split_name = get_split_name(scene, subject, args.split)
                if split_name is None:
                    continue

                rgb_dir = action_dir / "rgb"
                if not rgb_dir.exists():
                    missing_rgb_dirs.append(str(rgb_dir))
                    continue

                kp_full = load_rgb_keypoint_sequence(rgb_dir)  # [T, 17, 2]
                label = action_to_label[action]

                if args.mode == "full":
                    if kp_full.shape[0] < args.min_frames:
                        continue

                    frame_dir = f"{scene}_{subject}_{action}"
                    img_shape = infer_img_shape(
                        kp_full,
                        img_height=args.img_height,
                        img_width=args.img_width
                    )

                    anno = make_annotation(frame_dir, label, kp_full, img_shape)
                    annotations.append(anno)

                    split_dict[split_name].append(frame_dir)
                    if split_name == "val":
                        split_dict["test"].append(frame_dir)

                    length_counter[kp_full.shape[0]] += 1
                    split_counter[split_name] += 1
                    label_counter[label] += 1

                else:
                    key = (scene, subject, action)

                    if key not in segment_dict:
                        missing_segments.append(key)
                        continue

                    for seg_idx, (start_raw, end_raw) in enumerate(segment_dict[key]):
                        # Convert CSV frame number to zero-based Python index
                        start = start_raw - args.index_base
                        end = end_raw - args.index_base

                        if args.end_inclusive:
                            end = end + 1

                        start = max(0, start)
                        end = min(kp_full.shape[0], end)

                        if end <= start:
                            continue

                        kp_seg = kp_full[start:end]

                        if kp_seg.shape[0] < args.min_frames:
                            continue

                        frame_dir = f"{scene}_{subject}_{action}_seg{seg_idx:03d}"
                        img_shape = infer_img_shape(
                            kp_seg,
                            img_height=args.img_height,
                            img_width=args.img_width
                        )

                        anno = make_annotation(frame_dir, label, kp_seg, img_shape)
                        annotations.append(anno)

                        split_dict[split_name].append(frame_dir)
                        if split_name == "val":
                            split_dict["test"].append(frame_dir)

                        length_counter[kp_seg.shape[0]] += 1
                        split_counter[split_name] += 1
                        label_counter[label] += 1

    output = {
        "split": split_dict,
        "annotations": annotations
    }

    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    print("=" * 80)
    print(f"Saved PYSKL pkl to: {out_path}")
    print("=" * 80)

    print("\nNumber of annotations:", len(annotations))

    print("\nSplit counts:")
    for k, v in split_dict.items():
        print(f"  {k}: {len(v)}")

    print("\nLength distribution:")
    for length, count in length_counter.most_common(30):
        print(f"  T={length}: {count}")

    print("\nLabel distribution:")
    inv_label = {v: k for k, v in action_to_label.items()}
    for label, count in sorted(label_counter.items()):
        print(f"  label {label:02d} ({inv_label[label]}): {count}")

    if missing_rgb_dirs:
        print("\nMissing rgb dirs examples:")
        for x in missing_rgb_dirs[:10]:
            print(" ", x)

    if missing_segments:
        print("\nMissing segment keys examples:")
        for x in missing_segments[:10]:
            print(" ", x)

    print("\nAction-to-label mapping:")
    for action, label in action_to_label.items():
        print(f"  {action} -> {label}")

    print("\nDone.")


if __name__ == "__main__":
    main()