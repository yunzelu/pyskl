"""Protect identity alignment and unavailable-metric handling in OOF audits."""
import unittest

from rerun.pseudo_labeling.evaluate_oof_pseudo_labels import (
    align_to_annotations,
    saved_probabilities,
    summarize,
)


class TestOOFQualityEvaluation(unittest.TestCase):
    def test_join_uses_identity_instead_of_csv_order(self):
        rows = [
            {"source_pyskl_frame_dir": "second", "skeleton_sample_id": "s2"},
            {"source_pyskl_frame_dir": "first", "skeleton_sample_id": "s1"},
        ]
        aligned = align_to_annotations(rows, [{"frame_dir": "first"}, {"frame_dir": "second"}])
        self.assertEqual([row["skeleton_sample_id"] for row in aligned], ["s1", "s2"])

    def test_missing_extra_and_duplicate_windows_fail(self):
        row = {"source_pyskl_frame_dir": "first", "skeleton_sample_id": "s1"}
        for rows, annotations in (
            ([row], [{"frame_dir": "second"}]),
            ([row, row], [{"frame_dir": "first"}]),
            ([row], [{"frame_dir": "first"}, {"frame_dir": "first"}]),
        ):
            with self.subTest(rows=rows, annotations=annotations):
                with self.assertRaises(ValueError):
                    align_to_annotations(rows, annotations)

    def test_incomplete_probabilities_are_not_invented(self):
        complete = {f"mc_raw_p{i}": str(1 / 9) for i in range(9)}
        incomplete = dict(complete)
        incomplete["mc_raw_p3"] = ""
        self.assertIsNone(saved_probabilities([complete, incomplete], "mc_raw"))
        self.assertEqual(saved_probabilities([complete], "mc_raw").shape, (1, 9))

    def test_summary_keeps_metric_missing_from_first_fold(self):
        result = summarize([{"fold": "A"}, {"fold": "B", "nll": 1.0},
                            {"fold": "C", "nll": 3.0}])
        self.assertEqual(result[0]["metric"], "nll")
        self.assertEqual(result[0]["num_folds"], 2)
        self.assertEqual(result[0]["mean"], 2.0)
        self.assertAlmostEqual(result[0]["sd"], 2 ** 0.5)


if __name__ == "__main__":
    unittest.main()
