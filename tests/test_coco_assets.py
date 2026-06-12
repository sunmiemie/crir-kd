import tempfile
from pathlib import Path
import unittest
from test_image_headers import png_header
from crir_kd.coco_assets import inspect_coco_assets


class CocoAssetsTests(unittest.TestCase):
    def test_missing_mismatched_and_matching_assets(self):
        data = {"images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 24}], "categories": [], "annotations": []}
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(inspect_coco_assets(directory, data)["valid"])
            path = Path(directory) / "a.png"
            path.write_bytes(png_header(24, 32))
            self.assertFalse(inspect_coco_assets(directory, data)["valid"])
            path.write_bytes(png_header(32, 24))
            self.assertTrue(inspect_coco_assets(directory, data)["valid"])
