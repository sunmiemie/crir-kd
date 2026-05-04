import unittest

from crir_kd.boxes import Box
from crir_kd.diagnostics import size_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_boundary_and_unmatched_targets_are_retained(self):
        rows = [{"box": Box(0, 0, 2, 2), "matched": False, "quality": 0, "weight": .1},
                {"box": Box(0, 0, 4, 4), "matched": True, "quality": .8, "weight": .9}]
        report = size_diagnostics(rows, [4, 8], "student-input pixels")
        self.assertEqual([g["count"] for g in report["groups"]], [1, 1, 0])
        self.assertEqual(report["groups"][0]["unmatched_rate"], 1)
        self.assertEqual(report["groups"][0]["mean_weight"], .1)
        self.assertIsNone(report["groups"][2]["mean_weight"])
        self.assertFalse(report["official_dataset_metrics"])

    def test_invalid_boundaries_or_coordinate_space(self):
        for boundaries in [[4, 4], [8, 4], [0], [float("nan")]]:
            with self.assertRaises(ValueError):
                size_diagnostics([], boundaries, "original pixels")
        with self.assertRaises(ValueError):
            size_diagnostics([], [], "")
