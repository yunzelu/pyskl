"""Regression checks for MS-TCN segment matching without training dependencies."""

import unittest

import numpy as np

from rerun.e1.evaluate_continuous_segmental import (
    f1_counts_for_threshold,
    segments_from_labels,
)


def _segments(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return segments_from_labels(labels, np.arange(len(labels)), set(), "sequence_index")


def _reference_counts(ground_truth, prediction, threshold):
    """Independent vectorized form of the official best-match/hit-check rule.

    Reference: yabufarha/ms-tcn, eval.py at
    33ed91c0c7576650a2367efc602553af4c5295b1.
    Each prediction chooses its best GT before duplicate hits are rejected.
    Therefore TP equals the number of distinct eligible GT winners.
    """

    def runs(labels):
        labels = np.asarray(labels, dtype=np.int64)
        boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
        return labels[boundaries[:-1]], boundaries[:-1], boundaries[1:]

    gt_labels, gt_start, gt_end = runs(ground_truth)
    pred_labels, pred_start, pred_end = runs(prediction)
    intersection = np.minimum(pred_end[:, None], gt_end) - np.maximum(pred_start[:, None], gt_start)
    union = np.maximum(pred_end[:, None], gt_end) - np.minimum(pred_start[:, None], gt_start)
    scores = intersection / union * (pred_labels[:, None] == gt_labels)
    winners = np.argmax(scores, axis=1)
    eligible = scores[np.arange(len(pred_labels)), winners] >= threshold
    true_positives = len(np.unique(winners[eligible]))
    return true_positives, len(pred_labels) - true_positives, len(gt_labels) - true_positives


class TestContinuousSegmentalMetrics(unittest.TestCase):
    def test_duplicate_best_match_does_not_fall_back_to_unused_ground_truth(self):
        # GT: A[0,10), B[10,11), A[11,15).
        # Prediction: A[0,3), B[3,4), A[4,15).
        # The final A prefers the already-hit first A (IoU .4) over the
        # unused second A (IoU 4/11). It must count as an FP.
        ground_truth = [0] * 10 + [1] + [0] * 4
        prediction = [0] * 3 + [1] + [0] * 11
        for threshold in (0.10, 0.25):
            with self.subTest(threshold=threshold):
                counts = f1_counts_for_threshold(_segments(ground_truth), _segments(prediction), threshold)
                self.assertEqual(counts, (1, 2, 2))

    def test_overlap_equal_to_threshold_is_accepted(self):
        ground_truth = _segments([0, 0, 0, 0])
        prediction = _segments([0, 1, 1, 1])
        self.assertEqual(f1_counts_for_threshold(ground_truth, prediction, 0.25), (1, 1, 0))
        self.assertEqual(
            f1_counts_for_threshold(ground_truth, prediction, np.nextafter(0.25, 1.0)),
            (0, 2, 1),
        )

    def test_equal_iou_selects_first_ground_truth_segment(self):
        # A[2,7) has IoU 2/7 with both GT A runs. Choosing the first leaves
        # A[5,9) available to the later predicted A[8,9), whose IoU is .25.
        ground_truth = [0] * 4 + [1] + [0] * 4
        prediction = [1] * 2 + [0] * 5 + [1] + [0]
        counts = f1_counts_for_threshold(_segments(ground_truth), _segments(prediction), 0.25)
        self.assertEqual(counts, (2, 2, 1))

    def test_wrong_class_and_empty_segments(self):
        ground_truth = _segments([0, 0, 0])
        self.assertEqual(f1_counts_for_threshold(ground_truth, _segments([1, 1, 1]), 0.1), (0, 1, 1))
        self.assertEqual(f1_counts_for_threshold(ground_truth, [], 0.1), (0, 0, 1))
        self.assertEqual(f1_counts_for_threshold([], ground_truth, 0.1), (0, 1, 0))
        self.assertEqual(f1_counts_for_threshold([], [], 0.1), (0, 0, 0))

    def test_counts_match_official_reference_on_generated_sequences(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            ground_truth = np.repeat(rng.integers(0, 3, size=12), 6)
            prediction = np.roll(ground_truth, seed % 9 - 4)
            changed = rng.random(len(prediction)) < 0.12
            prediction[changed] = rng.integers(0, 3, size=np.count_nonzero(changed))
            for threshold in (0.10, 0.25, 0.50):
                with self.subTest(seed=seed, threshold=threshold):
                    expected = _reference_counts(ground_truth, prediction, threshold)
                    actual = f1_counts_for_threshold(_segments(ground_truth), _segments(prediction), threshold)
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
