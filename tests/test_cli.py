from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from crir_kd.__main__ import main


class CliTests(unittest.TestCase):
    def invoke(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            status = main(list(args))
        return status, output.getvalue(), error.getvalue()

    def test_synthetic_walkthrough_retains_unmatched_target(self):
        status, output, error = self.invoke("demo")
        report = json.loads(output)
        self.assertEqual(status, 0)
        self.assertEqual(error, "")
        self.assertTrue(report["synthetic"])
        self.assertFalse(report["training_implemented"])
        self.assertEqual(len(report["matches"]), 2)
        self.assertIsNone(report["matches"][1]["prediction_index"])
        self.assertGreater(report["toy_total_loss"], 2)

    def test_invalid_annotations_have_audit_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.txt"
            path.write_text("0 .9 .9 .4 .4", encoding="utf-8")
            status, output, error = self.invoke("yolo", str(path), "--classes", "1")
            self.assertEqual(status, 1)
            self.assertFalse(json.loads(output)["valid"])
            self.assertEqual(error, "")

    def test_bad_json_and_missing_file_are_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            status, _, error = self.invoke("coco", str(path))
            self.assertEqual(status, 2)
            self.assertIn("error", json.loads(error))
            path.write_text("{bad json", encoding="utf-8")
            status, _, error = self.invoke("coco", str(path))
            self.assertEqual(status, 2)
            self.assertNotIn("Traceback", error)

    def test_json_utf8_bom_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text('{"train": [{"path": "sample.jpg"}], "val": []}', encoding="utf-8-sig")
            status, output, _ = self.invoke("splits", str(path))
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output)["counts"]["train"], 1)
