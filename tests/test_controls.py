import unittest
from crir_kd.controls import reliability_controls


class ControlTests(unittest.TestCase):
    def test_mean_quality_includes_visible_misses(self):
        report = {"view_ids": ["a", "b"], "instances": [
            {"quality": 1, "stability": .5, "view_qualities": [0, None]},
            {"quality": 0, "stability": 0, "view_qualities": [0, 0]}]}
        result = reliability_controls(report)
        self.assertEqual(result["mean_view_quality"], [.5, 0])
        self.assertEqual(result["weights"]["joint_mean_control"], [.25, .25])
        self.assertEqual(result["teacher_forward_count"], 3)

    def test_view_budget_mismatch_fails(self):
        with self.assertRaises(ValueError):
            reliability_controls({"view_ids": [], "instances": [{"quality": 1, "stability": 1, "view_qualities": [1]}]})
