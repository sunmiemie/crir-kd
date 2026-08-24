import unittest
from crir_kd.permutation import permute_weights


class PermutationTests(unittest.TestCase):
    def test_reproducible_permutations_preserve_each_stratum(self):
        args = ([.1, .2, .8, .9], [2, 3, 12, 13], [.5] * 4, [8], [.4, .8])
        first = permute_weights(*args, seed=4)
        self.assertEqual(first, permute_weights(*args, seed=4))
        self.assertEqual(sorted(first["weights"][:2]), [.1, .2])
        self.assertEqual(sorted(first["weights"][2:]), [.8, .9])
        self.assertEqual(first["stratum_sizes"], [2, 2])

    def test_bad_boundaries_fail(self):
        with self.assertRaises(ValueError):
            permute_weights([], [], [], [8, 4], [])
