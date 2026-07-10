import unittest
from crir_kd.boxes import Box
from crir_kd.crops import CropView


class CropTests(unittest.TestCase):
    def test_partial_visibility_and_flip_inverse(self):
        view = CropView(Box(10, 20, 30, 40), 40, 40, True)
        box = Box(5, 25, 15, 35)
        visible = view.visibility(box)
        self.assertEqual(visible["fraction"], .5)
        self.assertEqual(view.transform.inverse(visible["box"]), Box(10, 25, 15, 35))
        self.assertFalse(view.visibility(box, .75)["visible"])
        self.assertFalse(view.visibility(Box(0, 0, 5, 5))["visible"])
