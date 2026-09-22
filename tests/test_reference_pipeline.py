import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from crir_kd.__main__ import main
from crir_kd.reference_pipeline import reference_pipeline


class ReferencePipelineTests(unittest.TestCase):
    def fixture(self):
        return json.loads((Path(__file__).parents[1] / "examples/reference.synthetic.json").read_text())

    def test_supplied_features_and_predictions_form_complete_reference(self):
        document = self.fixture()
        original = copy.deepcopy(document)
        result = reference_pipeline(document)
        self.assertEqual(result["instance_feature_losses"], [1])
        self.assertAlmostEqual(result["reference_total_losses"]["joint"], 1.45)
        self.assertEqual(result["reliability"]["instances"][0]["stability"], 1)
        self.assertEqual(document, original)

    def test_cli_runs_saved_example(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["reference", str(Path(__file__).parents[1] / "examples/reference.synthetic.json")])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["synthetic"])

    def test_missing_gt_feature_boxes_fail(self):
        document = self.fixture()
        document["features"]["student_boxes"] = []
        with self.assertRaises(ValueError):
            reference_pipeline(document)
