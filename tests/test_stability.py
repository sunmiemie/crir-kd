import unittest

from crir_kd.boxes import Box
from crir_kd.stability import instance_stability


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.box = Box(0, 0, 2, 2)

    def test_missing_visible_prediction_reduces_coverage(self):
        report = instance_stability(self.box, self.box, [self.box, None, None], [True, True, False])
        self.assertEqual(report["stability"], .5)
        self.assertEqual(report["missing_views"], 1)

    def test_invisible_views_do_not_count_as_misses(self):
        report = instance_stability(self.box, self.box, [self.box, None], [True, False])
        self.assertEqual(report["stability"], 1)

    def test_shifted_prediction_reduces_stability(self):
        shifted = Box(1, 0, 3, 2)
        report = instance_stability(self.box, self.box, [shifted], [True])
        self.assertAlmostEqual(report["mean_variation"], .125)
        self.assertLess(report["stability"], 1)

    def test_empty_missing_and_invisible_are_not_perfect(self):
        for reference, views, visible in [(None, [self.box], [True]), (self.box, [], []), (self.box, [None], [True]), (self.box, [self.box], [False])]:
            self.assertEqual(instance_stability(reference, self.box, views, visible)["stability"], 0)

    def test_bad_parameters_fail(self):
        with self.assertRaises(ValueError):
            instance_stability(self.box, self.box, [], [], tau=0)
        with self.assertRaises(ValueError):
            instance_stability(self.box, self.box, [None], [])
