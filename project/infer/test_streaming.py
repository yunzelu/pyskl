"""Checks for exact timestamp masking and bounded continuous-window input."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np

from project.infer import streaming


class StreamingInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.header = ["Timestamp", "ID", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"]
        self.header += [f"KP{index}_{axis}" for index in range(17) for axis in ("X", "Y", "C")]

    def write_csv(self, timestamps, ids=None):
        timestamps = list(timestamps)
        if ids is None:
            ids = [str(index + 100) for index in range(len(timestamps))]
        path = self.root / "pose_sample_ds_cf.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.header)
            for row_index, (timestamp, identity) in enumerate(zip(timestamps, ids)):
                values = [timestamp, identity, 0, 0, 1280, 720]
                for keypoint in range(17):
                    values.extend([row_index + keypoint, 2 * keypoint, "0.8"])
                writer.writerow(values)
        return path

    def write_mask(self, intervals=()):
        path = self.root / "pose_sample_ds_mm.json"
        path.write_text(json.dumps([
            {"start_unix_time": start, "end_unix_time": end}
            for start, end in intervals
        ]), encoding="utf-8")
        return streaming.read_mask(path)

    def frames(self, timestamps, intervals=(), ids=None, **sampling):
        stats = streaming.new_stats()
        result = list(streaming.iter_frames(
            self.write_csv(timestamps, ids), self.write_mask(intervals), stats, **sampling
        ))
        return result, stats

    def windows(self, timestamps, **options):
        frames, stats = self.frames(timestamps)
        result = list(streaming.iter_windows(iter(frames), stats, **options))
        return result, stats

    def test_mask_has_inclusive_endpoints_and_normalizes_union(self):
        mask = self.write_mask([(4, 5), (1, 2), (2, 3), (2.5, 4.5), (8, 8)])
        for timestamp in ("1", "2", "3", "4.5", "5", "8"):
            with self.subTest(timestamp=timestamp):
                self.assertTrue(mask.contains(Decimal(timestamp)))
        for timestamp in ("0.999999999", "5.000000001", "7.999999999", "8.000000001"):
            with self.subTest(timestamp=timestamp):
                self.assertFalse(mask.contains(Decimal(timestamp)))

    def test_missing_and_malformed_masks_fail(self):
        with self.assertRaises(FileNotFoundError):
            streaming.read_mask(self.root / "missing_mm.json")
        path = self.root / "invalid_mm.json"
        invalid = [
            {},
            [1],
            [{"start_unix_time": 1}],
            [{"start_unix_time": 2, "end_unix_time": 1}],
            [{"start_unix_time": float("nan"), "end_unix_time": 1}],
            [{"start_unix_time": 1, "end_unix_time": float("inf")}],
        ]
        for value in invalid:
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                streaming.read_mask(path)
        path.write_text("[unfinished", encoding="utf-8")
        with self.assertRaises(ValueError):
            streaming.read_mask(path)

    def test_exact_numeric_duplicate_timestamps_fail_even_for_same_id(self):
        for ids in (["4", "5"], ["4", "4"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.frames(["1", "1.00"], ids=ids)

    def test_duplicate_in_unsampled_phase_still_fails(self):
        with self.assertRaises(ValueError):
            self.frames(["0", "0.1", "0.10", "0.2", "0.3"], frame_step=3, phase=0)

    def test_exact_decimals_preserve_distinct_timestamps_and_original_text(self):
        times = ["1700000000.000000001", "1700000000.000000002", "1700000000.100000100"]
        frames, stats = self.frames(times)
        self.assertEqual([frame.timestamp for frame in frames], [Decimal(value) for value in times])
        self.assertEqual([frame.timestamp_text for frame in frames], times)
        self.assertEqual([frame.csv_row_number for frame in frames], [2, 3, 4])
        self.assertEqual(stats["retained_frames"], 3)

    def test_masked_multiperson_frames_are_removed_before_windows(self):
        frames, stats = self.frames(["0", "1", "1.0", "2", "2.00", "3"], [(1, 2)])
        self.assertEqual([frame.timestamp for frame in frames], [Decimal(0), Decimal(3)])
        self.assertEqual([frame.frame_index for frame in frames], [0, 3])
        self.assertEqual([frame.retained_index for frame in frames], [0, 1])
        self.assertEqual(stats["input_rows"], 6)
        self.assertEqual(stats["source_frames"], 4)
        self.assertEqual(stats["masked_rows"], 4)
        self.assertEqual(stats["masked_frames"], 2)
        self.assertEqual(stats["unmasked_frames"], 2)

    def test_phase_uses_source_timestamp_groups_before_masking(self):
        frames, stats = self.frames(
            ["0", "0.1", "0.10", "0.2", "0.3", "0.4", "0.5", "0.6"],
            [(0.1, 0.1), (0.3, 0.3)], frame_step=3, phase=0,
        )
        self.assertEqual([frame.frame_index for frame in frames], [0, 6])
        self.assertEqual([frame.timestamp for frame in frames], [Decimal(0), Decimal("0.6")])
        self.assertEqual([frame.retained_index for frame in frames], [0, 1])
        self.assertEqual(stats["source_frames"], 7)
        self.assertEqual(stats["masked_frames"], 2)
        self.assertEqual(stats["unmasked_frames"], 5)
        self.assertEqual(stats["sampling_dropped_frames"], 3)
        self.assertEqual(stats["retained_frames"], 2)

    def test_nonzero_sampling_phase(self):
        frames, _ = self.frames([Decimal(index) / 10 for index in range(8)], frame_step=3, phase=1)
        self.assertEqual([frame.frame_index for frame in frames], [1, 4, 7])

    def test_unsorted_nonfinite_and_invalid_timestamps_fail(self):
        for timestamps in (["1", "0.9"], ["NaN"], ["Infinity"], ["not a time"], [""]):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                self.frames(timestamps)

    def test_changing_ids_does_not_break_windows(self):
        windows, stats = self.windows([Decimal(index) / 10 for index in range(24)])
        self.assertEqual(len(windows), 2)
        self.assertEqual(len({frame.original_id for frame in windows[0].frames}), 20)
        self.assertEqual(stats["valid_windows"], 2)

    def test_window_size_stride_center_and_uncovered_tail(self):
        windows, stats = self.windows([Decimal(index) / 10 for index in range(29)])
        self.assertEqual([window.row_start for window in windows], [0, 4, 8])
        self.assertEqual([window.candidate_index for window in windows], [0, 1, 2])
        self.assertEqual([window.center.retained_index for window in windows], [10, 14, 18])
        self.assertEqual([len(window.frames) for window in windows], [20, 20, 20])
        self.assertEqual(stats["candidate_windows"], 3)
        self.assertEqual(stats["valid_windows"], 3)
        self.assertEqual(stats["uncovered_tail_frames"], 1)
        self.assertAlmostEqual(windows[0].max_gap_sec, 0.1)
        self.assertAlmostEqual(windows[0].span_sec, 1.9)

    def test_incomplete_input_produces_no_window(self):
        for length in (0, 1, 19):
            with self.subTest(length=length):
                windows, stats = self.windows([Decimal(index) / 10 for index in range(length)])
                self.assertEqual(windows, [])
                self.assertEqual(stats["candidate_windows"], 0)
                self.assertEqual(stats["uncovered_tail_frames"], length)

    def test_exact_timestamp_gap_and_span_limits_are_inclusive(self):
        timestamps = [Decimal(0), Decimal("0.5")]
        timestamps += [Decimal(index) / 10 for index in range(6, 23)]
        timestamps += [Decimal("2.5")]
        self.assertEqual(len(timestamps), 20)
        windows, stats = self.windows(timestamps)
        self.assertEqual(len(windows), 1)
        self.assertEqual(stats["dropped_validity_any"], 0)
        self.assertEqual(windows[0].max_gap_sec, 0.5)
        self.assertEqual(windows[0].span_sec, 2.5)

    def test_gap_span_rejection_reasons_count_independently(self):
        boundary = [Decimal(0), Decimal("0.5")]
        boundary += [Decimal(index) / 10 for index in range(6, 23)]
        boundary += [Decimal("2.5")]
        gap_only = list(boundary)
        gap_only[1] += Decimal("0.000000001")
        span_only = list(boundary)
        span_only[-1] += Decimal("0.000000001")
        both = [Decimal(index) / 10 + (Decimal("0.7") if index >= 10 else 0) for index in range(20)]
        for timestamps, gaps, spans in ((gap_only, 1, 0), (span_only, 0, 1), (both, 1, 1)):
            with self.subTest(gaps=gaps, spans=spans):
                windows, stats = self.windows(timestamps)
                self.assertEqual(windows, [])
                self.assertEqual(stats["candidate_windows"], 1)
                self.assertEqual(stats["dropped_max_adjacent_gap"], gaps)
                self.assertEqual(stats["dropped_max_window_span"], spans)
                self.assertEqual(stats["dropped_validity_any"], 1)

    def test_rejections_preserve_candidate_grid(self):
        times = [Decimal(index) / 10 + (Decimal(1) if index >= 4 else 0) for index in range(28)]
        windows, stats = self.windows(times)
        self.assertEqual([window.row_start for window in windows], [4, 8])
        self.assertEqual([window.candidate_index for window in windows], [1, 2])
        self.assertEqual(stats["candidate_windows"], 3)
        self.assertEqual(stats["dropped_validity_any"], 1)

    def test_chunking_preserves_overlap_and_centers(self):
        times = [Decimal(index) / 10 for index in range(57)]
        whole, whole_stats = self.windows(times)
        frames, chunked_stats = self.frames(times)
        chunks = list(streaming.iter_chunks(streaming.iter_windows(iter(frames), chunked_stats), 3))
        self.assertEqual([len(chunk) for chunk in chunks], [3, 3, 3, 1])
        flattened = [window for chunk in chunks for window in chunk]
        signature = lambda windows: [
            (window.row_start, window.center.timestamp, tuple(frame.frame_index for frame in window.frames))
            for window in windows
        ]
        self.assertEqual(signature(flattened), signature(whole))
        self.assertEqual(chunked_stats, whole_stats)

    def test_annotation_preserves_single_person_pose_and_confidence(self):
        windows, _ = self.windows([Decimal(index) / 10 for index in range(20)])
        annotation = windows[0].to_annotation()
        self.assertEqual(annotation["total_frames"], 20)
        self.assertEqual(annotation["keypoint"].shape, (1, 20, 17, 2))
        self.assertEqual(annotation["keypoint_score"].shape, (1, 20, 17))
        np.testing.assert_array_equal(annotation["keypoint"][0, 10, 5], [15, 10])
        np.testing.assert_allclose(annotation["keypoint_score"], 0.8)


if __name__ == "__main__":
    unittest.main()
