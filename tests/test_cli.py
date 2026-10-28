from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

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

    def test_box_svg_invalid_annotation_containers_are_input_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boxes.json"
            for annotations in (None, False, True, 0, 1, 1.5, "", "label", {}, {"xywh": [1, 2, 3, 4]}):
                with self.subTest(annotations=annotations):
                    content = json.dumps({"width": 10, "height": 10, "annotations": annotations})
                    path.write_text(content, encoding="utf-8")
                    status, output, error = self.invoke("box-svg", str(path))
                    self.assertEqual(status, 2)
                    self.assertEqual(output, "")
                    self.assertEqual(json.loads(error), {"error": "annotations must be an array of objects"})
                    self.assertEqual(path.read_text(encoding="utf-8"), content)
                    self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_box_svg_accepts_missing_empty_and_populated_annotations(self):
        cases = ({}, {"annotations": []}, {"annotations": [{"xywh": [1, 2, 3, 4], "label": "<target>&"}]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boxes.json"
            for fields in cases:
                with self.subTest(fields=fields):
                    path.write_text(json.dumps({"width": 10, "height": 10, **fields}), encoding="utf-8")
                    status, output, error = self.invoke("box-svg", str(path))
                    self.assertEqual(status, 0)
                    self.assertEqual(error, "")
                    root = ET.fromstring(output)
                    self.assertEqual(root.attrib["viewBox"], "0 0 10 10")
                    rectangles = root.findall("{http://www.w3.org/2000/svg}rect")
                    self.assertEqual(len(rectangles), 1 + len(fields.get("annotations", [])))
                    if fields.get("annotations"):
                        self.assertEqual(rectangles[1].attrib["width"], "3")
                        self.assertEqual(rectangles[1].attrib["height"], "4")
                        self.assertEqual(root.find("{http://www.w3.org/2000/svg}text").text, "<target>&")

    def test_box_svg_bad_later_row_does_not_emit_partial_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boxes.json"
            path.write_text(json.dumps({"width": 10, "height": 10,
                                        "annotations": [{"xywh": [1, 2, 3, 4]}, None]}), encoding="utf-8")
            status, output, error = self.invoke("box-svg", str(path))
            self.assertEqual(status, 2)
            self.assertEqual(output, "")
            self.assertIn("annotations", json.loads(error)["error"])
