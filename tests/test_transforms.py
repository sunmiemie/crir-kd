import unittest

from crir_kd.boxes import Box
from crir_kd.transforms import ViewTransform


class TransformTests(unittest.TestCase):
    def test_letterbox_accounts_for_padding(self):
        transform = ViewTransform.letterbox((640, 320), (640, 640))
        self.assertEqual(transform.forward(Box(10, 20, 30, 40)), Box(10, 180, 30, 200))

    def test_inverse_with_flip_and_nonuniform_scale(self):
        transform = ViewTransform(0.5, 0.25, 10, 12, 400, True)
        original = Box(2, 4, 12, 24)
        self.assertEqual(transform.inverse(transform.forward(original)), original)

    def test_feature_coordinates(self):
        transform = ViewTransform.letterbox((100, 100), (200, 200))
        self.assertEqual(transform.to_feature(Box(10, 20, 30, 40), 4), Box(5, 10, 15, 20))

    def test_invalid_metadata(self):
        for sizes in [((0, 1), (2, 2)), ((2, 2), (-1, 2))]:
            with self.assertRaises(ValueError):
                ViewTransform.letterbox(*sizes)
        with self.assertRaises(ValueError):
            ViewTransform(1, 1, 0, 0, 10).to_feature(Box(0, 0, 1, 1), 0)
