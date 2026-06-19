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
