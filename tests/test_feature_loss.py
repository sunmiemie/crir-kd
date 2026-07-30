import unittest
from crir_kd.boxes import Box
from crir_kd.feature_loss import feature_mse, local_feature_losses


class FeatureLossTests(unittest.TestCase):
    def test_mse_and_normalization_control(self):
        self.assertEqual(feature_mse([[[1, 2]]], [[[2, 4]]]), 2.5)
        self.assertAlmostEqual(feature_mse([[[1, 2]]], [[[2, 4]]], True), 0)
        self.assertEqual(feature_mse([[[0]]], [[[0]]], True), 0)

    def test_cross_resolution_constant_features_and_empty_gt(self):
        self.assertEqual(local_feature_losses([[[1]]], [[[1, 1], [1, 1]]], [Box(0, 0, 1, 1)], [Box(0, 0, 2, 2)]), [0])
        self.assertEqual(local_feature_losses([], [], [], []), [])

    def test_invalid_channels_and_overflow(self):
        with self.assertRaises(ValueError):
            feature_mse([[[1]]], [[[1]], [[2]]])
        with self.assertRaises(ValueError):
            feature_mse([1], [1])
        with self.assertRaises(ValueError):
            feature_mse([[[1e308]]], [[[-1e308]]])
