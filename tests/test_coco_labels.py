import copy
import unittest

from crir_kd.coco_labels import inspect_coco


class CocoLabelsTests(unittest.TestCase):
    def setUp(self):
        self.data = {"images": [{"id": 1, "width": 100, "height": 50, "file_name": "x.jpg"}],
                     "categories": [{"id": 3, "name": "car"}],
                     "annotations": [{"id": 5, "image_id": 1, "category_id": 3, "bbox": [1, 2, 3, 4]}]}

    def test_sparse_category_ids_are_valid(self):
        original = copy.deepcopy(self.data)
        self.assertTrue(inspect_coco(self.data)["valid"])
        self.assertEqual(original, self.data)

    def test_duplicate_ids_and_missing_references(self):
        self.data["images"].append(self.data["images"][0])
        self.data["annotations"][0]["image_id"] = 99
        result = inspect_coco(self.data)
        self.assertFalse(result["valid"])
        self.assertEqual(len(result["issues"]), 2)

    def test_bad_boxes_and_crowd_values(self):
        for patch in [{"bbox": [0, 0, float("nan"), 1]}, {"bbox": [-1, 0, 1, 1]}, {"iscrowd": 2}, {"bbox": "bad"}]:
            data = copy.deepcopy(self.data)
            data["annotations"][0].update(patch)
            self.assertFalse(inspect_coco(data)["valid"])

    def test_segmentation_area_not_required_to_equal_box_area(self):
        self.data["annotations"][0]["area"] = 5
        self.assertTrue(inspect_coco(self.data)["valid"])

    def test_ignore_accepts_missing_boolean_and_binary_integer_flags(self):
        for fields in ({}, {"ignore": 0}, {"ignore": 1}, {"ignore": False}, {"ignore": True}):
            with self.subTest(fields=fields):
                data = copy.deepcopy(self.data)
                data["annotations"][0].update(fields)
                original = copy.deepcopy(data)
                report = inspect_coco(data)
                self.assertTrue(report["valid"])
                self.assertEqual(report["valid_annotation_count"], 1)
                self.assertEqual(report["issues"], [])
                self.assertEqual(data, original)

    def test_invalid_ignore_flags_report_the_annotation_without_mutating_input(self):
        for value in (None, "0", "1", "false", "", [], [0], {}, {"value": 0}, -1, 2, 0.0, 1.0, 0.5):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["annotations"].append({**data["annotations"][0], "id": 6, "ignore": value})
                original = copy.deepcopy(data)
                report = inspect_coco(data)
                self.assertFalse(report["valid"])
                self.assertEqual(report["valid_annotation_count"], 1)
                self.assertEqual(len(report["issues"]), 1)
                issue = report["issues"][0]
                self.assertEqual((issue["section"], issue["index"], issue["severity"]), ("annotations", 1, "error"))
                self.assertIn("ignore", issue["message"])
                self.assertEqual(data, original)

    def test_bad_top_level_structure(self):
        with self.assertRaises(ValueError):
            inspect_coco({"images": []})
