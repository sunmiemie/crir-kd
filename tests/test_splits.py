import unittest

from crir_kd.splits import inspect_splits


class SplitTests(unittest.TestCase):
    def test_different_paths_with_equal_hash_reveal_leakage(self):
        result = inspect_splits({"train": [{"path": "a.jpg", "sha256": "a" * 64}],
                                 "val": [{"path": "b.jpg", "sha256": "a" * 64}]})
        self.assertEqual(result["issues"][0]["kind"], "duplicate_content")
        self.assertTrue(result["issues"][0]["cross_split"])

    def test_source_video_crosses_splits(self):
        result = inspect_splits({"train": [{"path": "a.jpg", "source_group": "video1"}],
                                 "val": [{"path": "b.jpg", "source_group": "video1"}]})
        self.assertEqual(result["issues"][0]["kind"], "shared_source_group")

    def test_fingerprint_is_order_independent(self):
        first = inspect_splits({"train": [{"path": "b.jpg"}, {"path": "a.jpg"}], "val": []})
        second = inspect_splits({"val": [], "train": [{"path": "a.jpg"}, {"path": "b.jpg"}]})
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertTrue(first["valid"])

    def test_invalid_paths_and_hashes(self):
        for row in [{"path": "../a.jpg"}, {"path": "C:/a.jpg"}, {"path": "/a.jpg"}, {"path": "a.jpg", "sha256": "x"}]:
            with self.assertRaises(ValueError):
                inspect_splits({"train": [row]})
