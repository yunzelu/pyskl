# Related-project dataset

Build the first 10 fps training dataset from existing YOLO26x-pose predictions:

```powershell
python project/dataset/build_radar_v4_10fps.py
```

Run from the repository root with Python 3.8+ and NumPy installed. Use the
training environment to keep NumPy pickle compatibility. No GPU or new pose
inference is needed. The builder imports the label map, recording selection,
and continuous-window logic from `rerun/dataset/`. The train/validation split
and single-PKL writer are defined in the project builder.

The initial artifacts were built with the local `har` environment (Python
3.8.20, NumPy 1.24.4):

```powershell
conda run --no-capture-output -n har python project/dataset/build_radar_v4_10fps.py
```

## Inputs and outputs

- Pose input: `data/radar_v4/raw_jsonl/yolo26xpose/<recording>/*.jsonl`
- Camera times: `data/radar_v4/origin/<recording>/timestamps.csv`
  (`timestamp.csv` is also accepted when it is the only matching file).
- Default output: `data/project/yolo26xpose/fps10_phase0/`
- Detected JSONLs: `detected_jsonl/<recording>/*__fps10_phase0__detected-only.jsonl`
- PKL: `pyskl/continuous_window_w20_s4/`
- Audit counts, label map, class/subject/split counts: `stats/`

The pickle filename is:

```text
radarv4_yolo26xpose_continuous_window_w20_s4_val_yunze.pkl
```

It contains `annotations` and `split` with only `train` and `val`:

| Split | Subjects | Windows |
| --- | --- | --- |
| Train | chenzhe, dengdeng, han, hui, jiadi, li, mia, rose, saad, xilai | 54,492 |
| Validation | yunze | 4,968 |

All recordings for a subject belong to the same split. Keypoints have shape
`(1, 20, 17, 2)` and confidence scores have shape `(1, 20, 17)`. This build uses hard center
labels. All new code lives in `project/`; generated data is ignored by Git
through the existing `/data` rule.

## Sampling and real timestamps

1. Select original camera frames satisfying `frame_idx % 3 == 0`: frames
   `0, 3, 6, ...`. Phase is anchored to recording frame 0, before detection
   filtering. It does not restart after gaps or at annotation boundaries.
2. Remove selected rows where `detected` is exactly `false`, as in rerun.
3. Join each frame to the CSV using `Frame == frame_idx`. Set `timestamp_sec`
   to the CSV `Timestamp` value, in Unix seconds. Preserve the original
   calculated time as `source_timestamp_sec`; add `elapsed_timestamp_sec`
   relative to CSV frame 0. Original frame indices, skeletons, frame labels,
   and annotation segment boundaries are preserved.
4. Slide over retained detected rows using a **20-row window, stride 4**.
   Drop incomplete tail windows. The hard label is from row `start + 10`
   (the 11th retained frame), following rerun's even-window center convention.
5. Keep a window only when the maximum adjacent real timestamp gap is
   **<= 0.5 seconds** and its first-to-last real timestamp span is
   **<= 2.5 seconds**. Reject centers outside the final nine classes.

Ten fps is nominal: selecting every third camera frame preserves real capture
jitter and pauses. No timestamp resampling, interpolation, or padding is used.
A regular 20-frame window spans about 1.9 seconds between its endpoints and
advances about 0.4 seconds per stride. PKL `timestamps_sec` uses float64 Unix
seconds from the JSONL, with original frame indices retained for tracing.

Missing CSV matches, duplicate frame keys, non-finite times, and timestamps
that do not increase with frame index cause an error rather than falling back
to `frame_idx / 30`. Matching uses actual JSONL frame indices; the video
container's reported frame count is not used for alignment.

The source pose run's `process_summary` is stored as metadata
`source_process_summary`. Project row counts and sampling/timestamp provenance
are in `preprocess_info`.

Only `sit`, `fall`, and `laysofa` recording families, including numbered repeats,
are included. Standalone `walk` recordings are excluded, as in rerun.

## Fixed labels

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

Aliases are inherited directly from rerun, including LayBed/LayFloor -> lie,
Walking -> walk, and Falling -> fall.

## Options and checks

Input/output roots can be overridden with `--raw-jsonl-root`, `--origin-root`,
and `--output-root`. The initial build uses only phase 0. Future phases can be
built separately with `--phase 1` or `--phase 2`; their default output roots
are `fps10_phase1` and `fps10_phase2`. Each invocation rebuilds its JSONLs and
the PKL from the raw inputs.

```powershell
python -m unittest project.dataset.test_build_radar_v4_10fps
```

Training configuration creation is a separate step. A project config should
point to this PKL and use `clip_len=20`; the existing rerun configs resample
to 60 frames and refer to the thesis data paths.

## Phase-0 build audit

The build produced 43 detected JSONLs containing 263,625 frames and
59,460 distinct windows in one PKL, with yunze reserved for validation.
Every retained JSONL frame was compared with its original pose record and
CSV timestamp. Every saved annotation was checked against the JSONLs for
frame indices, poses, scores, timestamps, center label, and both validity
rules. Train/validation assignments were checked for complete coverage and disjoint
subjects. Results are saved in the output's `stats/verification.json`.

For `49-jiadi-laysofa`, the existing raw pose JSONL contains 18,056 frame rows
while the CSV contains 18,490 timestamps. The build uses the available pose
rows and reports the 434 unmatched CSV rows in its preprocessing statistics
and audit report.
