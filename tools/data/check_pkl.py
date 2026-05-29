import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def summarize_array(name, arr):
    if arr is None:
        print(f"{name}: None")
        return

    print(f"{name}:")
    print(f"  type: {type(arr)}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min: {np.nanmin(arr):.4f}")
    print(f"  max: {np.nanmax(arr):.4f}")
    print(f"  mean: {np.nanmean(arr):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        type=str,
        required=True,
        help="Path to ntu120_hrnet.pkl"
    )
    args = parser.parse_args()

    pkl_path = Path(args.pkl)

    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    print("=" * 80)
    print(f"Loading: {pkl_path}")
    print("=" * 80)

    data = load_pkl(pkl_path)

    print("\n[1] Top-level keys")
    print(data.keys())

    if "split" not in data:
        raise KeyError("This pickle file does not contain key: 'split'")

    if "annotations" not in data:
        raise KeyError("This pickle file does not contain key: 'annotations'")

    split = data["split"]
    annotations = data["annotations"]

    print("\n[2] Split names and sample counts")
    for split_name, video_ids in split.items():
        print(f"{split_name}: {len(video_ids)} samples")

    print("\n[3] Total annotations")
    print(f"Number of annotation items: {len(annotations)}")

    if len(annotations) == 0:
        raise ValueError("annotations is empty")

    print("\n[4] First annotation keys")
    first = annotations[0]
    print(first.keys())

    identifier = "filename" if "filename" in first else "frame_dir"
    print(f"\nIdentifier used by PYSKL: {identifier}")

    print("\n[5] First annotation basic information")
    print(f"{identifier}: {first.get(identifier)}")
    print(f"label: {first.get('label')}")
    print(f"total_frames: {first.get('total_frames')}")
    print(f"img_shape: {first.get('img_shape')}")
    print(f"original_shape: {first.get('original_shape')}")

    print("\n[6] First annotation array shapes")
    keypoint = first.get("keypoint", None)
    keypoint_score = first.get("keypoint_score", None)

    summarize_array("keypoint", keypoint)
    summarize_array("keypoint_score", keypoint_score)

    print("\n[7] Check whether split IDs exist in annotations")
    all_annotation_ids = set(item[identifier] for item in annotations)

    for split_name, video_ids in split.items():
        video_id_set = set(video_ids)
        missing = video_id_set - all_annotation_ids
        extra_annotations = all_annotation_ids - video_id_set

        print(f"\nSplit: {split_name}")
        print(f"  IDs in split: {len(video_id_set)}")
        print(f"  Missing from annotations: {len(missing)}")
        print(f"  Annotations not in this split: {len(extra_annotations)}")

        if len(missing) > 0:
            print("  Example missing IDs:")
            for x in list(missing)[:5]:
                print(f"    {x}")

    print("\n[8] Label distribution for each split")
    id_to_anno = {item[identifier]: item for item in annotations}

    for split_name, video_ids in split.items():
        labels = []
        for vid in video_ids:
            if vid in id_to_anno:
                labels.append(id_to_anno[vid]["label"])

        counter = Counter(labels)

        print(f"\nSplit: {split_name}")
        print(f"  Number of labels: {len(labels)}")
        print(f"  Number of classes: {len(counter)}")
        print(f"  Min label: {min(counter.keys())}")
        print(f"  Max label: {max(counter.keys())}")

        most_common = counter.most_common(5)
        least_common = sorted(counter.items(), key=lambda x: x[1])[:5]

        print("  Top 5 most common classes:")
        for label, count in most_common:
            print(f"    class {label}: {count}")

        print("  Top 5 least common classes:")
        for label, count in least_common:
            print(f"    class {label}: {count}")

    print("\n[9] Check keypoint shape consistency")
    shape_counter = Counter()
    score_shape_counter = Counter()
    frame_mismatch_count = 0

    for item in annotations:
        kp = item.get("keypoint", None)
        ks = item.get("keypoint_score", None)

        if kp is not None:
            shape_counter[kp.shape] += 1

            # PYSKL format: [M, T, V, C]
            if "total_frames" in item:
                if kp.shape[1] != item["total_frames"]:
                    frame_mismatch_count += 1

        if ks is not None:
            score_shape_counter[ks.shape] += 1

    print("\nMost common keypoint shapes:")
    for shape, count in shape_counter.most_common(10):
        print(f"  {shape}: {count}")

    print("\nMost common keypoint_score shapes:")
    for shape, count in score_shape_counter.most_common(10):
        print(f"  {shape}: {count}")

    print(f"\nNumber of samples where keypoint.shape[1] != total_frames: {frame_mismatch_count}")

    print("\n[10] Check whether any samples have M > 1")

    multi_person_items = []
    m_counter = Counter()

    for item in annotations:
        kp = item.get("keypoint", None)
        if kp is None:
            continue

        # Expected shape: [M, T, V, C]
        if len(kp.shape) != 4:
            print(f"Warning: unexpected keypoint shape for {item.get(identifier)}: {kp.shape}")
            continue

        M, T, V, C = kp.shape
        m_counter[M] += 1

        if M > 1:
            multi_person_items.append({
                "id": item.get(identifier),
                "label": item.get("label"),
                "shape": kp.shape,
                "total_frames": item.get("total_frames")
            })

    print("\nM dimension distribution:")
    for m, count in sorted(m_counter.items()):
        print(f"  M={m}: {count} samples")

    print(f"\nNumber of samples with M > 1: {len(multi_person_items)}")

    if len(multi_person_items) > 0:
        print("\nExamples of samples with M > 1:")
        for x in multi_person_items[:20]:
            print(
                f"  id={x['id']}, "
                f"label={x['label']}, "
                f"shape={x['shape']}, "
                f"total_frames={x['total_frames']}"
            )
    else:
        print("No samples with M > 1 were found.")

    print("\nDone.")


if __name__ == "__main__":
    main()