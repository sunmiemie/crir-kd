import unittest
from crir_kd.boxes import Box
from crir_kd.crops import CropView
from crir_kd.matching import Instance, Prediction
from crir_kd.multiview import analyze_views


class MultiViewTests(unittest.TestCase):
    def test_inverse_coordinates_and_invisible_targets(self):
        first, second = Box(1, 1, 3, 3), Box(11, 1, 13, 3)
        gt = [Instance(first, 0), Instance(second, 0)]
        crop = CropView(Box(0, 0, 10, 10), 20, 20, True)
        report = analyze_views(gt, [Prediction(first, 0, .9)],
                               [{"id": "crop", "view": crop, "predictions": [Prediction(crop.transform.forward(first), 0, .8)]}])
        self.assertEqual(report["instances"][0]["stability"], 1)
        self.assertEqual(report["instances"][1]["visible_views"], 0)
        self.assertEqual(report["instances"][1]["view_qualities"], [None])
        self.assertEqual(report["teacher_forward_count"], 2)

    def test_visible_miss_stays_in_denominator(self):
        box = Box(1, 1, 3, 3)
        view = CropView(Box(0, 0, 10, 10), 10, 10)
        report = analyze_views([Instance(box, 0)], [Prediction(box, 0, 1)], [{"id": "miss", "view": view, "predictions": []}])
        self.assertEqual(report["instances"][0]["missing_views"], 1)
        self.assertEqual(report["instances"][0]["stability"], 0)
