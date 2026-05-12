import unittest
import sys

from crir_kd.weighting import matched_mean_control, reliability_weights, total_loss, weighted_kd_loss


class WeightingTests(unittest.TestCase):
    def test_joint_floor_and_ablation_strategies(self):
        weights = reliability_weights([0, 1], [1, .5], .2)
        self.assertAlmostEqual(weights[0], .2)
        self.assertAlmostEqual(weights[1], .6)
        self.assertEqual(reliability_weights([.3], [.7], strategy="quality"), [.3])
        self.assertEqual(reliability_weights([.3], [.7], strategy="stability"), [.7])
        self.assertEqual(reliability_weights([0], [0], strategy="uniform"), [1])

    def test_zero_weights_preserve_detection_supervision(self):
        self.assertEqual(total_loss(3.5, [2, 4], [0, 0], 2), 3.5)
        self.assertEqual(total_loss(3.5, [], [], 2), 3.5)

    def test_denominator_is_instance_count(self):
        self.assertEqual(weighted_kd_loss([2, 4], [.5, .5]), 1.5)

    def test_mean_control_matches_strength(self):
        self.assertEqual(matched_mean_control([0, .5, 1]), [.5, .5, .5])
        self.assertEqual(matched_mean_control([]), [])

    def test_large_finite_terms_have_a_representable_mean(self):
        self.assertEqual(weighted_kd_loss([1e308, 1e308], [1, 1]), 1e308)
        self.assertEqual(weighted_kd_loss([sys.float_info.max] * 3, [1] * 3), sys.float_info.max)
        self.assertEqual(weighted_kd_loss([1e308, 1e308, 0], [1, 1, 1]), (2 / 3) * 1e308)

    def test_zero_strength_preserves_detection_with_large_instance_losses(self):
        self.assertEqual(total_loss(3.5, [1e308, 1e308], [1, 1], 0), 3.5)

    def test_unrepresentable_total_loss_is_a_clear_input_error(self):
        for call in [lambda: total_loss(1, [1e308], [1], 2),
                     lambda: total_loss(1e308, [1e308], [1], 1)]:
            with self.subTest(call=call), self.assertRaisesRegex(ValueError, "total_loss.*finite"):
                call()

    def test_invalid_values_and_lengths(self):
        for call in [lambda: reliability_weights([2], [0]), lambda: weighted_kd_loss([1], []),
                     lambda: total_loss(-1, [], [], 1), lambda: weighted_kd_loss([-1], [1])]:
            with self.assertRaises(ValueError):
                call()
