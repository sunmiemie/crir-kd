import copy
import unittest
from crir_kd.conversion import coco_to_yolo
from crir_kd.yolo_labels import inspect_yolo_labels


class ConversionTests(unittest.TestCase):
    def document(self):
        return {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50}, {"id": 2, "file_name": "b.jpg", "width": 100, "height": 50}],
                "categories": [{"id": 9, "name": "car"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 9, "bbox": [40, 15, 20, 20]}]}

    def test_sparse_categories_round_trip_and_negative_images(self):
        report = coco_to_yolo(self.document())
        self.assertEqual(report["class_mapping"][0]["class_id"], 0)
        self.assertEqual(report["images"][0]["label_text"], "0 0.5 0.5 0.2 0.4\n")
        self.assertTrue(inspect_yolo_labels(report["images"][0]["label_text"], 1)["valid"])
        self.assertEqual(report["images"][1]["label_text"], "")

    def test_crowd_requires_explicit_lossy_policy(self):
        data = self.document()
        data["annotations"][0]["iscrowd"] = 1
        with self.assertRaises(ValueError):
            coco_to_yolo(data)
        self.assertEqual(coco_to_yolo(data, "skip")["skipped_annotation_ids"], [1])

    def test_ignore_flags_preserve_labels_or_require_explicit_skip(self):
        for value in (0, False, 1, True):
            with self.subTest(value=value):
                data = self.document()
                data["annotations"].append({**data["annotations"][0], "id": 2, "ignore": value})
                original = copy.deepcopy(data)
                if value:
                    with self.assertRaisesRegex(ValueError, "Crowd/ignored"):
                        coco_to_yolo(data)
                else:
                    self.assertEqual(coco_to_yolo(data), coco_to_yolo(data, "skip") | {"ignored_policy": "reject"})
                report = coco_to_yolo(data, "skip")
                self.assertEqual(report["skipped_annotation_ids"], [2] if value else [])
                self.assertEqual(report["images"][0]["label_text"], "0 0.5 0.5 0.2 0.4\n" * (1 if value else 2))
                self.assertEqual(report["images"][1]["label_text"], "")
                self.assertEqual(data, original)

    def test_invalid_ignore_flags_fail_validation_for_both_conversion_policies(self):
        for value in (None, "0", "1", "false", "", [], [0], {}, {"value": 0}, -1, 2, 0.0, 1.0, 0.5):
            for policy in ("reject", "skip"):
                with self.subTest(value=value, policy=policy):
                    data = self.document()
                    data["annotations"].append({**data["annotations"][0], "id": 2, "ignore": value})
                    original = copy.deepcopy(data)
                    with self.assertRaisesRegex(ValueError, "COCO structure failed validation"):
                        coco_to_yolo(data, policy)
                    self.assertEqual(data, original)
