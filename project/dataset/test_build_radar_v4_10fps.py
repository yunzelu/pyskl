"""Regression checks for timestamp alignment, sampling phase, and window rules."""

from __future__ import annotations

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.dataset import build_radar_v4_10fps as project


class ProjectDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_csv(self, rows):
        path = self.root / "timestamps.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Frame", "Timestamp"])
            writer.writerows(rows)
        return path

    def source_jsonl(self):
        path = self.root / "2-saad-sit" / "poses.jsonl"
        path.parent.mkdir(exist_ok=True)
        metadata = {
            "type": "metadata",
            "dataset_info": {"subject": "saad", "session_name": "2-saad-sit"},
            "video_info": {"height": 480, "width": 640, "assumed_fps_used_for_timestamp": 30},
            "annotation_info": {"segments": [{"start_frame": 0, "end_frame": 71, "label": "Walking"}]},
        }
        frames = [{
            "type": "frame", "frame_idx": index, "timestamp_sec": index / 30,
            "detected": index not in {1, 3}, "label": "Walking",
            "keypoints_xy": [[index, 2.0]] * 17, "keypoints_conf": [0.8] * 17,
        } for index in range(72)]
        with path.open("w", encoding="utf-8") as handle:
            for record in [metadata] + frames + [{"type": "process_summary", "frames_processed": 72}]:
                handle.write(json.dumps(record) + "\n")
        return path, metadata, frames

    def session(self, timestamps, subject="saad"):
        count = len(timestamps)
        identity = project.rerun.parse_session_dir_name(f"2-{subject}-sit")
        return project.rerun.SessionData(
            identity=identity, jsonl_path=self.root / identity.directory_name / "poses.jsonl",
            raw_jsonl_path="raw.jsonl", img_shape=(480, 640),
            keypoint=np.ones((count, 17, 2), dtype=np.float32),
            keypoint_score=np.ones((count, 17), dtype=np.float32),
            frame_indices=np.arange(count, dtype=np.int32) * 3,
            timestamps_sec=np.asarray(timestamps, dtype=np.float64),
            frame_labels=["Walking"] * count, segments=[], num_keypoints=17,
        )

    def test_phase_is_camera_index_and_csv_is_joined_by_frame(self):
        source, original_metadata, original_frames = self.source_jsonl()
        source_bytes = source.read_bytes()
        times = {index: 1736614256.0 + index * 0.035 for index in range(72)}
        # Reversed CSV order ensures timestamps are joined by Frame, not row position.
        csv_path = self.write_csv(reversed(list(times.items())))
        for phase in range(3):
            with self.subTest(phase=phase):
                output = self.root / f"phase{phase}" / "2-saad-sit" / "poses.jsonl"
                stats = project.preprocess_session(source, csv_path, output, phase)
                metadata, frames = project.rerun.read_processed_jsonl(output)
                expected = [i for i in range(72) if i % 3 == phase and i not in {1, 3}]
                self.assertEqual([f["frame_idx"] for f in frames], expected)
                self.assertEqual(metadata["annotation_info"], original_metadata["annotation_info"])
                self.assertNotIn("assumed_fps_used_for_timestamp", metadata["video_info"])
                self.assertEqual(metadata["source_process_summary"]["frames_processed"], 72)
                self.assertEqual(stats["kept_frame_rows"], len(expected))
                for frame in frames:
                    index = frame["frame_idx"]
                    self.assertEqual(frame["timestamp_sec"], times[index])
                    self.assertEqual(frame["elapsed_timestamp_sec"], times[index] - times[0])
                    self.assertEqual(frame["source_timestamp_sec"], index / 30)
                    self.assertEqual(frame["keypoints_xy"], original_frames[index]["keypoints_xy"])
                result = project.build_project_windows([project.rerun.load_session(output)], phase)
                for annotation in result.annotations:
                    np.testing.assert_array_equal(annotation["timestamps_sec"],
                                                  [times[i] for i in annotation["source_frame_indices"]])
        self.assertEqual(source.read_bytes(), source_bytes)

    def test_invalid_csv_timestamps_fail(self):
        cases = [
            [(0, 1.0), (0, 1.1)],
            [(0, 1.0), (1, float("nan"))],
            [(0, 1.0), (1, float("inf"))],
            [(0, 1.0), (1, 0.9)],
            [(0, 1.0), (1, 1.0)],
            [(1, 1.0)],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                project.read_timestamps(self.write_csv(rows))

    def test_missing_csv_frame_does_not_fall_back_to_assumed_fps(self):
        source, _, _ = self.source_jsonl()
        csv_path = self.write_csv((i, 1700000000 + i / 30) for i in range(72) if i != 6)
        output = self.root / "output.jsonl"
        with self.assertRaisesRegex(ValueError, "Missing timestamp for Frame 6"):
            project.preprocess_session(source, csv_path, output)
        self.assertFalse(output.exists())

    def test_window_size_stride_center_alias_and_tail(self):
        session = self.session(np.arange(29) / 10)
        session.frame_labels[10] = "Transition-LayBed-to-Sit"
        result = project.build_project_windows([session], 0)
        self.assertEqual([a["window_row_start"] for a in result.annotations], [0, 4, 8])
        first = result.annotations[0]
        self.assertEqual(first["total_frames"], 20)
        self.assertEqual(first["keypoint"].shape, (1, 20, 17, 2))
        self.assertEqual(first["keypoint_score"].shape, (1, 20, 17))
        self.assertEqual(first["center_source_frame"], 30)
        self.assertEqual(first["label"], 4)
        self.assertEqual(first["label_name"], "transition-lie-to-sit")
        self.assertEqual(result.stats["candidate_windows"], 3)

    def test_both_timestamp_rules_and_inclusive_thresholds(self):
        gap_only = np.arange(20) * 0.01
        gap_only[10:] += 0.59
        boundary_gap = np.arange(20) * 0.1
        boundary_gap[10:] += 0.4
        cases = [
            (np.arange(20) * 0.1, None),
            (gap_only, "max_adjacent_gap"),
            (np.arange(20) * 0.14, "max_window_span"),
            (boundary_gap, None),
            (np.linspace(0, 2.5, 20), None),
        ]
        for timestamps, reason in cases:
            with self.subTest(reason=reason, span=timestamps[-1]):
                result = project.build_project_windows([self.session(timestamps)], 0)
                if reason is None:
                    self.assertEqual(len(result.annotations), 1)
                else:
                    self.assertEqual(len(result.annotations), 0)
                    self.assertEqual(result.stats["dropped_windows_by_reason"], {reason: 1, "validity_any": 1})

    def test_invalid_center_label_is_excluded(self):
        session = self.session(np.arange(20) / 10)
        session.frame_labels[10] = "DELETE"
        result = project.build_project_windows([session], 0)
        self.assertFalse(result.annotations)
        self.assertEqual(result.stats["dropped_windows_by_reason"]["center_label_not_in_final_set"], 1)

    def test_saved_split_holds_out_only_yunze_without_subject_leakage(self):
        training_subjects = {"chenzhe", "dengdeng", "han", "hui", "jiadi", "li", "mia", "rose", "saad", "xilai"}
        subjects = training_subjects | {"yunze"}
        sessions = [self.session(np.arange(24) / 10, subject) for subject in sorted(subjects)]
        result = project.build_project_windows(sessions, 0)
        path = project.save_project_protocol(self.root, result)
        self.assertEqual(list(self.root.rglob("*.pkl")), [path])
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        annotations = {a["frame_dir"]: a for a in payload["annotations"]}
        self.assertEqual(set(payload["split"]), {"train", "val"})
        all_ids = [name for ids in payload["split"].values() for name in ids]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(annotations))
        self.assertEqual({annotations[name]["subject"] for name in payload["split"]["train"]}, training_subjects)
        self.assertEqual({annotations[name]["subject"] for name in payload["split"]["val"]}, {"yunze"})
        self.assertEqual(len(payload["split"]["train"]), 20)
        self.assertEqual(len(payload["split"]["val"]), 2)
        summary = json.loads(path.with_name(f"{path.stem}_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["num_samples_by_split"], {"train": 20, "val": 2})
        self.assertEqual(summary["subjects"]["val"], ["yunze"])

    def test_split_rejects_missing_subjects_and_duplicate_sample_ids(self):
        subjects = {s for assigned in project.SPLIT_SUBJECTS.values() for s in assigned}
        for invalid in (subjects - {"yunze"}, subjects - {"mia"}, subjects | {"unknown"}):
            with self.subTest(subjects=invalid), self.assertRaises(ValueError):
                project.validate_subjects(invalid)
        sessions = [self.session(np.arange(20) / 10, subject) for subject in sorted(subjects)]
        result = project.build_project_windows(sessions, 0)
        with self.assertRaisesRegex(ValueError, "Duplicate sample IDs"):
            project.make_project_split(result.annotations + [result.annotations[0]])


if __name__ == "__main__":
    unittest.main()
