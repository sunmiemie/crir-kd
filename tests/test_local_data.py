import tempfile
from pathlib import Path
import unittest

from crir_kd.local_data import dataset_path, hash_splits


class LocalDataTests(unittest.TestCase):
    def test_hashes_actual_content_and_finds_renamed_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"same")
            (root / "b").write_bytes(b"same")
            result = hash_splits(root, {"train": [{"path": "a"}], "val": [{"path": "b"}]})
            self.assertFalse(result["valid"])
            self.assertEqual(result["issues"][0]["kind"], "duplicate_content")
            with self.assertRaises(ValueError):
                hash_splits(root, {"train": [{"path": "a", "sha256": "0" * 64}]})

    def test_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ("../outside", "/outside", "C:/outside", ""):
                with self.assertRaises(ValueError):
                    dataset_path(directory, path)
