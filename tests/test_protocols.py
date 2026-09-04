import copy
import unittest
from crir_kd.protocols import protocol_fingerprint, compare_protocols


def protocol_fixture():
    return {"dataset": {"version": "synthetic-v1", "split_id": "toy-training"},
            "student": {"implementation": "toy-grid", "input_size": [16, 16]},
            "teacher": {"checkpoint_id": "handwritten-features", "input_size": [32, 32]},
            "optimization": {"epochs": 1}, "augmentation": {}, "evaluation": {"implementation": "toy-mse"}, "kd_features": {"layer": "grid"}}


class ProtocolTests(unittest.TestCase):
    def test_order_invariance_and_controlled_difference(self):
        first = protocol_fixture()
        self.assertEqual(protocol_fingerprint(first), protocol_fingerprint(dict(reversed(list(first.items())))))
        second = copy.deepcopy(first)
        second["student"]["input_size"] = [32, 32]
        report = compare_protocols(first, second)
        self.assertFalse(report["comparable"])
        self.assertEqual(report["differences"][0]["path"], "student.input_size")

    def test_method_and_nonfinite_settings_rejected(self):
        first = protocol_fixture()
        first["method"] = "joint"
        with self.assertRaises(ValueError):
            protocol_fingerprint(first)
        del first["method"]
        first["optimization"]["lr"] = float("nan")
        with self.assertRaises(ValueError):
            protocol_fingerprint(first)
