import copy
import json
import math
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

    def test_type_changes_inside_arrays_report_the_original_array_path(self):
        cases = (
            ("optimization", "lr_schedule", "multiplier",
             [{"epoch": 0, "multiplier": 1}], [{"epoch": 0, "multiplier": 1.0}]),
            ("augmentation", "transforms", "enabled",
             [{"name": "horizontal_flip", "enabled": False}],
             [{"name": "horizontal_flip", "enabled": 0}]),
        )
        for section, key, value_key, left, right in cases:
            with self.subTest(path=f"{section}.{key}"):
                first, second = protocol_fixture(), protocol_fixture()
                first[section][key], second[section][key] = left, right
                original = json.dumps([first, second])
                report = compare_protocols(first, second)
                self.assertFalse(report["comparable"])
                self.assertNotEqual(report["left_fingerprint"], report["right_fingerprint"])
                self.assertEqual(report["differences"], [{"path": f"{section}.{key}", "left": left, "right": right}])
                difference = report["differences"][0]
                self.assertIs(type(difference["left"][0][value_key]), type(left[0][value_key]))
                self.assertIs(type(difference["right"][0][value_key]), type(right[0][value_key]))
                self.assertEqual(json.dumps([first, second]), original)

    def test_dictionary_key_order_inside_arrays_is_not_a_difference(self):
        first, second = protocol_fixture(), protocol_fixture()
        first["optimization"]["lr_schedule"] = [{"epoch": 0, "multiplier": 1}]
        second["optimization"]["lr_schedule"] = [{"multiplier": 1, "epoch": 0}]
        report = compare_protocols(first, second)
        self.assertTrue(report["comparable"])
        self.assertEqual(report["left_fingerprint"], report["right_fingerprint"])
        self.assertEqual(report["differences"], [])

    def test_signed_zero_difference_explains_changed_fingerprint(self):
        first, second = protocol_fixture(), protocol_fixture()
        first["optimization"]["momentum"] = -0.0
        second["optimization"]["momentum"] = 0.0
        report = compare_protocols(first, second)
        self.assertFalse(report["comparable"])
        self.assertEqual(report["differences"], [{"path": "optimization.momentum", "left": -0.0, "right": 0.0}])
        self.assertEqual(math.copysign(1, report["differences"][0]["left"]), -1)
        self.assertEqual(math.copysign(1, report["differences"][0]["right"]), 1)
