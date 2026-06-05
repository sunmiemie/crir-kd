import tempfile
from pathlib import Path
import unittest

from test_image_headers import png_header
from crir_kd.yolo_dataset import inspect_yolo_dataset


class YoloDatasetTests(unittest.TestCase):
    def test_audit_maps_boxes_to_pixels_and_distinguishes_missing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(png_header(100, 50))
            (root / "a.txt").write_text("0 .5 .5 .2 .4", encoding="utf-8")
            report = inspect_yolo_dataset(root, [{"image": "a.png", "label": "a.txt"}], 1)
            self.assertTrue(report["valid"])
            self.assertAlmostEqual(report["images"][0]["boxes"][0]["xywh"][2], 20)
            (root / "a.txt").write_text("", encoding="utf-8")
            self.assertTrue(inspect_yolo_dataset(root, [{"image": "a.png", "label": "a.txt"}], 1)["images"][0]["negative_image"])
            self.assertFalse(inspect_yolo_dataset(root, [{"image": "a.png", "label": "missing.txt"}], 1)["valid"])
