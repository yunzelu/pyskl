"""Fusion and streamed inference output checks without loading model weights."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np

from project.infer import infer_csv


def one_hot_streams(count):
    result = {}
    for index, stream in enumerate(("j", "b", "jm", "bm")):
        values = np.zeros((count, 9), dtype=np.float64)
        values[:, index] = 1
        result[stream] = values
    return result


class FakePredictor:
    def __init__(self):
        self.batch_lengths = []
        self.center_values = []

    def predict(self, annotations):
        self.batch_lengths.append(len(annotations))
        self.center_values.extend(float(item["keypoint"][0, 10, 0, 0]) for item in annotations)
        return one_hot_streams(len(annotations))

    def describe(self):
        return {"test_predictor": True}


class FusionTests(unittest.TestCase):
    def test_weighted_probabilities_are_averaged_without_another_softmax(self):
        streams = one_hot_streams(2)
        # Different predictions in the second row ensure fusion preserves order.
        for values in streams.values():
            values[1] = values[0, ::-1]
        actual = infer_csv.fuse_probabilities(streams)
        expected = np.array([1 / 3, 1 / 3, 1 / 6, 1 / 6, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(actual[0], expected, rtol=0, atol=0)
        np.testing.assert_allclose(actual[1], expected[::-1], rtol=0, atol=0)
        # A second softmax would give positive mass to these absent classes.
        np.testing.assert_array_equal(actual[0, 4:], np.zeros(5))

    def test_identical_streams_preserve_the_original_distribution(self):
        original = np.array([[0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 0]])
        streams = {stream: original.copy() for stream in infer_csv.WEIGHTS}
        np.testing.assert_allclose(infer_csv.fuse_probabilities(streams), original)

    def test_exactly_four_expected_stream_keys_are_required(self):
        missing = one_hot_streams(1)
        del missing["bm"]
        extra = one_hot_streams(1)
        extra["other"] = extra["j"]
        for streams in (missing, extra, {}):
            with self.subTest(keys=list(streams)), self.assertRaises(ValueError):
                infer_csv.fuse_probabilities(streams)

    def test_invalid_shapes_and_batch_mismatches_fail(self):
        for bad in (np.ones(9) / 9, np.ones((1, 8)) / 8, np.ones((2, 9)) / 9):
            streams = one_hot_streams(1)
            streams["bm"] = bad
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                infer_csv.fuse_probabilities(streams)

    def test_invalid_probability_values_fail(self):
        invalid = [
            [float("nan"), 0, 0, 0, 0, 0, 0, 0, 0],
            [float("inf"), 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.1, 1.1, 0, 0, 0, 0, 0, 0, 0],
            [0.2, 0.2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for values in invalid:
            streams = one_hot_streams(1)
            streams["jm"] = np.array([values])
            with self.subTest(values=values), self.assertRaises(ValueError):
                infer_csv.fuse_probabilities(streams)


class InferenceOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.output_dir = self.root / "output"
        self.output_dir.mkdir()

    def write_pair(self, name="pose_2026-06-23_ds_cf.csv", count=29, timestamps=None, directory=None):
        directory = directory or self.input_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        mask = directory / (name.removesuffix("_cf.csv") + "_mm.json")
        header = ["Timestamp", "ID", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"]
        header += [f"KP{index}_{axis}" for index in range(17) for axis in ("X", "Y", "C")]
        if timestamps is None:
            timestamps = [format(Decimal("1700000000.000000001") + Decimal(index) / 10, "f")
                          for index in range(count)]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index, timestamp in enumerate(timestamps):
                writer.writerow([timestamp, f"track{100 + index}", 0, 0, 1280, 720]
                                + [value for keypoint in range(17)
                                   for value in (index + keypoint, keypoint, 0.9)])
        mask.write_text("[]\n", encoding="utf-8")
        return path, mask

    def args(self, **overrides):
        values = dict(output_dir=self.output_dir, chunk_size=2, max_windows=None, validate_only=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def process(self, path, mask, args=None, predictor=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return infer_csv.process_one(path, mask, args or self.args(), predictor)

    def test_find_inputs_pairs_masks_and_excludes_april(self):
        april, _ = self.write_pair("pose_2026-04-18_ds_cf.csv", count=0)
        june, june_mask = self.write_pair("pose_2026-06-23_ds_cf.csv", count=0)
        july, july_mask = self.write_pair("pose_2026-07-20_ds_cf.csv", count=0)
        (self.input_dir / "pose_raw.csv").write_text("unused", encoding="utf-8")
        pairs = infer_csv.find_inputs([self.input_dir, june], ["*2026-04-18*"])
        self.assertEqual(pairs, [(june.resolve(), june_mask.resolve()), (july.resolve(), july_mask.resolve())])
        self.assertNotIn(april.resolve(), [path for path, _ in pairs])

    def test_find_inputs_requires_matching_mask(self):
        path, mask = self.write_pair(count=0)
        mask.unlink()
        with self.assertRaises(FileNotFoundError):
            infer_csv.find_inputs([path], [])

    def test_find_inputs_rejects_colliding_basenames(self):
        first, _ = self.write_pair(count=0)
        second, _ = self.write_pair(count=0, directory=self.root / "second")
        with self.assertRaises(ValueError):
            infer_csv.find_inputs([first, second], [])

    def test_find_inputs_rejects_empty_selection(self):
        self.write_pair("pose_2026-04-18_ds_cf.csv", count=0)
        with self.assertRaises(ValueError):
            infer_csv.find_inputs([self.input_dir], ["*2026-04-18*"])

    def test_predictions_keep_exact_center_provenance_and_chunk_overlap(self):
        path, mask = self.write_pair()
        input_bytes, mask_bytes = path.read_bytes(), mask.read_bytes()
        predictor = FakePredictor()
        summary = self.process(path, mask, predictor=predictor)
        self.assertEqual(predictor.batch_lengths, [2, 1])
        self.assertEqual(predictor.center_values, [10, 14, 18])
        with Path(summary["prediction_csv"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["window_start_retained_idx"] for row in rows], ["0", "4", "8"])
        self.assertEqual([row["window_candidate_index"] for row in rows], ["0", "1", "2"])
        self.assertEqual([row["source_csv_row_center"] for row in rows], ["12", "16", "20"])
        self.assertEqual([row["source_frame_center"] for row in rows], ["10", "14", "18"])
        self.assertEqual([row["source_id"] for row in rows], ["track110", "track114", "track118"])
        self.assertEqual([row["ID"] for row in rows], ["0", "0", "0"])
        self.assertEqual(rows[0]["center_timestamp_sec"], "1700000001.000000001")
        self.assertEqual(rows[0]["label_id"], "0")
        self.assertEqual(rows[0]["label_name"], infer_csv.FINAL_LABELS[0])
        self.assertEqual(float(rows[0]["confidence"]), 1 / 3)
        self.assertEqual(float(rows[0]["j_p0"]), 1)
        self.assertEqual(float(rows[0]["fused_p8"]), 0)
        self.assertEqual(summary["counts"]["uncovered_tail_frames"], 1)
        self.assertEqual(summary["output_rows"], 3)
        self.assertEqual(summary["chunks_processed"], 2)
        self.assertTrue(summary["scan_complete"])
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["model"], {"test_predictor": True})
        saved_summary = json.loads((self.output_dir / f"{path.stem}__inference_summary.json").read_text())
        self.assertEqual(saved_summary, summary)
        self.assertEqual(path.read_bytes(), input_bytes)
        self.assertEqual(mask.read_bytes(), mask_bytes)
        self.assertEqual(list(self.output_dir.glob("*.partial")), [])

    def test_duplicate_after_first_chunk_does_not_publish_partial_results(self):
        timestamps = [str(Decimal(index) / 10) for index in range(29)]
        timestamps.append("2.800")
        path, mask = self.write_pair(timestamps=timestamps)
        original = path.read_bytes()
        prediction_path = self.output_dir / f"{path.stem}__center_predictions.csv"
        summary_path = self.output_dir / f"{path.stem}__inference_summary.json"
        for prior_output in (None, "previous successful output\n"):
            with self.subTest(prior_output=prior_output):
                if prior_output is not None:
                    prediction_path.write_text(prior_output, encoding="utf-8")
                predictor = FakePredictor()
                with self.assertRaises(ValueError):
                    self.process(path, mask, predictor=predictor)
                self.assertEqual(predictor.batch_lengths, [2])
                if prior_output is None:
                    self.assertFalse(prediction_path.exists())
                else:
                    self.assertEqual(prediction_path.read_text(), prior_output)
                self.assertFalse(summary_path.exists())
                self.assertEqual(list(self.output_dir.glob("*.partial")), [])
                self.assertEqual(path.read_bytes(), original)
                self.assertTrue(mask.exists())

    def test_validate_only_scans_without_predictor_or_prediction_csv(self):
        path, mask = self.write_pair()
        result = self.process(path, mask, self.args(validate_only=True))
        self.assertEqual(result["mode"], "validation")
        self.assertEqual(result["counts"]["valid_windows"], 3)
        self.assertEqual(result["output_rows"], 0)
        self.assertIsNone(result["prediction_csv"])
        self.assertIsNone(result["model"])
        self.assertEqual(list(self.output_dir.glob("*.csv*")), [])
        self.assertTrue((self.output_dir / f"{path.stem}__validation_summary.json").exists())

    def test_max_windows_marks_summary_as_a_limited_scan(self):
        path, mask = self.write_pair(count=45)
        predictor = FakePredictor()
        summary = self.process(path, mask, self.args(max_windows=3), predictor)
        self.assertEqual(summary["status"], "limited")
        self.assertFalse(summary["scan_complete"])
        self.assertEqual(summary["max_windows"], 3)
        self.assertEqual(summary["output_rows"], 3)
        self.assertEqual(summary["counts"]["valid_windows"], 3)
        self.assertLess(summary["counts"]["input_rows"], 45)
        self.assertEqual(predictor.batch_lengths, [2, 1])


if __name__ == "__main__":
    unittest.main()
