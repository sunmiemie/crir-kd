import unittest

from crir_kd.yolo_labels import inspect_yolo_labels


class YoloLabelsTests(unittest.TestCase):
    def test_empty_negative_image_is_valid(self):
        self.assertTrue(inspect_yolo_labels("\n", 2)["valid"])

    def test_valid_row_and_duplicate_warning(self):
        result = inspect_yolo_labels("0 .5 .5 .2 .4\n0 .5 .5 .2 .4", 2)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["labels"]), 2)
        self.assertEqual(result["issues"][0]["severity"], "warning")

    def test_collect_errors_with_line_numbers(self):
        result = inspect_yolo_labels("2 .5 .5 .2 .2\n0 .9 .9 .4 .4\n0 nan .5 .2 .2\n0 .5 .5 -.1 .2\n1.0 .5 .5 .2 .2", 2)
        self.assertFalse(result["valid"])
        self.assertEqual([issue["line"] for issue in result["issues"]], [1, 2, 3, 4, 5])

    def test_reject_invalid_class_count(self):
        with self.assertRaises(ValueError):
            inspect_yolo_labels("", True)
