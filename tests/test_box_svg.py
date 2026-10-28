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

    def test_annotations_require_an_array_including_empty_containers(self):
        for annotations in (None, False, True, 0, 1, 1.5, "", "label", {}, {"xywh": [1, 2, 3, 4]}):
            with self.subTest(annotations=annotations):
                with self.assertRaisesRegex(ValueError, "annotations must be an array of objects"):
                    box_svg(10, 10, annotations)

    def test_empty_and_populated_tuples_match_list_output(self):
        for annotations in ([], [{"xywh": [1, 2, 3, 4], "label": "target"}]):
            with self.subTest(annotations=annotations):
                self.assertEqual(box_svg(10, 10, annotations), box_svg(10, 10, tuple(annotations)))
