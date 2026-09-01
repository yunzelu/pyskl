# RADAR v4 YOLO26x-pose Rerun Dataset Protocol

This folder contains the new dataset-building code for the thesis rerun. The
builder is intentionally separate from the older `tools/data/radar_v4` scripts
so the rerun protocol is explicit and uniform.

## Script

```powershell
python rerun/dataset/build_radar_v4_yolo26xpose_datasets.py
```

Default inputs and outputs:

- Raw JSONL input: `data/radar_v4/raw_jsonl/yolo26xpose`
- Rerun artifacts: `data/radar_v4/rerun/yolo26xpose`
- Detected-only JSONLs: `data/radar_v4/rerun/yolo26xpose/detected_jsonl`
- PYSKL pkls: `data/radar_v4/rerun/yolo26xpose/pyskl`
- Sidecar stats: `data/radar_v4/rerun/yolo26xpose/stats`

Use `--skip-preprocess` to reuse already generated detected-only JSONLs.

## Recording Selection

Session folder names are parsed as:

```text
<recording-index>-<subject>-<session>
```

Examples:

- `5-han-laysofa2`: index `5`, subject `han`, session `laysofa2`
- `43-rose-fall`: index `43`, subject `rose`, session `fall`

Only session families `fall`, `sit`, and `laysofa` are used. Numeric suffixes
are treated as repeated recordings of the same family, so `fall2`, `sit2`,
`laysofa2`, and `laysofa3` are included. Standalone `-walk` recordings are
excluded.

## Preprocessing

Each source JSONL is rewritten to a detected-only JSONL by removing frame rows
where `detected` is exactly `false`. The original `frame_idx` and
`timestamp_sec` values are preserved. Metadata segments in
`annotation_info/segments` are preserved and a `preprocess_info` block is added
to the metadata line.

## Labels

All labels are lowercased before mapping. Final label IDs are fixed in this
order:

| ID | Label |
| --- | --- |
| 0 | `lie-stationary` |
| 1 | `sit-stationary` |
| 2 | `walk` |
| 3 | `fall` |
| 4 | `transition-lie-to-sit` |
| 5 | `transition-lie-to-stand` |
| 6 | `transition-sit-to-lie` |
| 7 | `transition-sit-to-stand` |
| 8 | `transition-stand-to-sit` |

Mapping aliases:

- `LayFloor-Stationary`, `LayBed-Stationary` -> `lie-stationary`
- `Walking` -> `walk`
- `Falling` -> `fall`
- `Transition-LayFloor-to-Sit`, `Transition-LayBed-to-Sit` -> `transition-lie-to-sit`
- `Transition-LayFloor-to-Stand`, `Transition-LayBed-to-Stand` -> `transition-lie-to-stand`
- `Transition-Sit-to-LayFloor`, `Transition-Sit-to-LayBed` -> `transition-sit-to-lie`

Labels outside the final label set, such as `DELETE` and kneeling labels, are
not emitted as samples.

## Dataset Protocols

Activity-aligned samples:

- Use only `annotation_info/segments` from the JSONL metadata line.
- Each mapped manual segment becomes one sample.
- Exclude the trailing open-ended interval from recordings known not to end
  with an `END` annotation: `12-xilai-sit2`, `19-saad-laysofa`, and
  `7-han-sit`.
- Require at least 2 detected skeleton rows inside the annotated segment.
- Require maximum adjacent timestamp gap <= 0.5 seconds inside the detected
  skeleton rows.
- No total-duration/span constraint is applied.
- No context expansion, interpolation, fixed-length clipping, or zero-fill rule
  is applied.
- Because preprocessing removes no-detection rows, sample length is the number
  of detected skeleton rows inside the annotated segment.

Continuous-window samples:

- Use detected skeleton rows after preprocessing.
- Window size: 60 rows.
- Stride: 12 rows.
- Incomplete tail windows are dropped.
- Center label comes from row `start + 30`.
- A window is valid only if:
  - maximum adjacent timestamp gap <= 0.5 seconds
  - first-to-last timestamp span <= 2.5 seconds

Windows whose center label is outside the final label set are not emitted.

Continuous-window triangular temporal-composition samples:

- Use exactly the same windows, validity checks, hard center labels, and splits
  as `continuous_window_w60_s12`.
- Add `label_soft_triangular` beside the hard center `label`.
- For each retained frame position, use the original `frame_idx` to find the
  matching `annotation_info/segments` label.
- Use normalized symmetric Bartlett weights:
  `w_t proportional to 1 - abs(2*t - (L - 1)) / (L - 1)`.
- For `L=60`, positions 0 and 59 have zero mass and positions 29 and 30 share
  the two equal center peaks.
- Frames whose labels are outside the final nine classes, such as `DELETE`,
  kneeling labels, or no covering segment, are ignored.
- The remaining valid triangular mass is renormalized, so
  `sum(label_soft_triangular) == 1`.
- Because the center frame must have a valid final label, zero valid target
  mass is treated as an error.

## Folds

The same subject-wise folds are written for all protocols:

| Fold | Train | Validation | Calibration | Test |
| --- | --- | --- | --- | --- |
| A | chenzhe, dengdeng, hui, jiadi, mia, rose, xilai, yunze | han | saad | li |
| B | han, jiadi, li, mia, rose, saad, xilai, yunze | hui | chenzhe | dengdeng |
| C | chenzhe, dengdeng, han, hui, jiadi, li, saad, xilai | rose | mia | yunze |

Pkl names:

- `radarv4_yolo26xpose_activity_aligned_fold_a.pkl`
- `radarv4_yolo26xpose_activity_aligned_fold_b.pkl`
- `radarv4_yolo26xpose_activity_aligned_fold_c.pkl`
- `radarv4_yolo26xpose_continuous_window_w60_s12_fold_a.pkl`
- `radarv4_yolo26xpose_continuous_window_w60_s12_fold_b.pkl`
- `radarv4_yolo26xpose_continuous_window_w60_s12_fold_c.pkl`
- `radarv4_yolo26xpose_continuous_window_w60_s12_triangular_fold_a.pkl`
- `radarv4_yolo26xpose_continuous_window_w60_s12_triangular_fold_b.pkl`
- `radarv4_yolo26xpose_continuous_window_w60_s12_triangular_fold_c.pkl`

Each pkl follows the PYSKL format:

```python
{
    "split": {"train": [...], "val": [...], "calib": [...], "test": [...]},
    "annotations": [...]
}
```

## Stats

Sidecar stats are written under `data/radar_v4/rerun/yolo26xpose/stats`:

- `preprocess_jsonl.csv`
- `preprocess_summary.json`
- `label_map.csv`
- `<protocol>/samples_by_subject.csv`
- `<protocol>/samples_by_class.csv`
- `<protocol>/samples_by_subject_class.csv`
- `<protocol>/samples_by_fold_split.csv`
- `<protocol>/samples_by_fold_split_class.csv`
- `<protocol>/samples_by_fold_split_subject.csv`
- `<protocol>/summary.json`
