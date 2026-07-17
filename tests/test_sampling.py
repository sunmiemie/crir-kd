import unittest
from crir_kd.boxes import Box
from crir_kd.sampling import sample_box


class SamplingTests(unittest.TestCase):
    def test_center_interpolation_and_identity_sampling(self):
        features = [[[0, 2], [4, 6]]]
        self.assertEqual(sample_box(features, Box(0, 0, 2, 2), (1, 1)), [[[3]]])
        self.assertEqual(sample_box(features, Box(0, 0, 2, 2)), features)

    def test_cross_resolution_linear_fields_align(self):
        low = [[[.5, 1.5], [.5, 1.5]]]
        high = [[[(x + .5) / 2 for x in range(4)] for _ in range(4)]]
        self.assertEqual(sample_box(low, Box(0, 0, 2, 2)), sample_box(high, Box(0, 0, 4, 4)))

    def test_outside_and_nonrectangular_inputs_fail(self):
        with self.assertRaises(ValueError):
            sample_box([[[1], [2, 3]]], Box(0, 0, 1, 1))
        with self.assertRaises(ValueError):
            sample_box([[[1]]], Box(-1, 0, 1, 1))
