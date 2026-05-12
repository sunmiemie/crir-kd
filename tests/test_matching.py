import unittest

from crir_kd.boxes import Box
from crir_kd.matching import Instance, Prediction, match_quality


class MatchingTests(unittest.TestCase):
    def test_quality_uses_iou_and_confidence(self):
        result = match_quality([Instance(Box(0, 0, 2, 2), 0)], [Prediction(Box(0, 0, 2, 1), 0, .8)])
        self.assertAlmostEqual(result[0]["quality"], .4)

    def test_prediction_cannot_supervise_two_targets(self):
        gt = [Instance(Box(0, 0, 2, 2), 0), Instance(Box(0, 0, 2, 2), 0)]
        result = match_quality(gt, [Prediction(Box(0, 0, 2, 2), 0, .9)])
        self.assertEqual([r["prediction_index"] for r in result], [0, None])

    def test_wrong_class_and_missing_predictions_remain_unmatched(self):
        gt = [Instance(Box(0, 0, 2, 2), 0)]
        self.assertEqual(match_quality(gt, [Prediction(gt[0].box, 1, 1)])[0]["quality"], 0)
        self.assertIsNone(match_quality(gt, [])[0]["prediction_index"])

    def test_disjoint_boxes_cannot_match(self):
        with self.assertRaises(ValueError):
            match_quality([], [], min_iou=0)

    def test_bad_prediction_score_rejected(self):
        with self.assertRaises(ValueError):
            Prediction(Box(0, 0, 1, 1), 0, float("nan"))

    def test_large_valid_boxes_do_not_become_unmatched(self):
        box = Box(0, 0, 1e154, 1e154)
        result = match_quality([Instance(box, 0)], [Prediction(box, 0, .75)])
        self.assertEqual(result[0]["prediction_index"], 0)
        self.assertEqual(result[0]["iou"], 1)
        self.assertEqual(result[0]["quality"], .75)
