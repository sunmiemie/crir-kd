import unittest
import xml.etree.ElementTree as ET
from crir_kd.box_svg import box_svg


class BoxSvgTests(unittest.TestCase):
    def test_escapes_labels_and_preserves_pixel_geometry(self):
        svg = box_svg(100, 50, [{"xywh": [10, 20, 4, 5], "label": "<script>&test"}])
        root = ET.fromstring(svg)
        self.assertEqual(root.attrib["viewBox"], "0 0 100 50")
        self.assertNotIn("<script>", svg)
        self.assertIn('width="4" height="5"', svg)

    def test_outside_boxes_are_reported(self):
        with self.assertRaises(ValueError):
            box_svg(10, 10, [{"xywh": [9, 9, 2, 2]}])
