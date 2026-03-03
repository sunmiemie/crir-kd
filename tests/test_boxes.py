import unittest

from crir_kd.boxes import Box, iou


class BoxTests(unittest.TestCase):
    def test_conversion_and_continuous_area(self):
        box = Box.from_xywh(2, 3, 4, 5)
        self.assertEqual(box.as_xywh(), (2, 3, 4, 5))
        self.assertEqual(box.area, 20)
        self.assertEqual(box.center, (4, 5.5))

    def test_overlap_and_touching_edges(self):
        a = Box(0, 0, 2, 2)
        self.assertEqual(iou(a, a), 1)
        self.assertEqual(iou(a, Box(2, 0, 3, 1)), 0)
        self.assertAlmostEqual(iou(a, Box(1, 1, 3, 3)), 1 / 7)

    def test_clip_retains_visible_region(self):
        self.assertEqual(Box(-1, -2, 3, 5).clip(2, 4), Box(0, 0, 2, 4))
        self.assertIsNone(Box(4, 4, 5, 5).clip(3, 3))

    def test_invalid_boxes_fail(self):
        for args in [(0, 0, 0, 1), (2, 0, 1, 2), (0, 0, float("nan"), 1), (False, 0, 1, 2),
                     (0, 0, 10 ** 400, 1), (0, 0, 1e300, 1e300), (0, 0, 1e-300, 1e-300)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Box(*args)
        with self.assertRaises(ValueError):
            Box.from_xywh(0, 0, -1, 2)
