import unittest

from crir_kd.provenance import REQUIRED_TEXT, validate_run_record


class ProvenanceTests(unittest.TestCase):
    def valid_record(self):
        record = {field: "synthetic fixture" for field in REQUIRED_TEXT}
        record.update(code_commit="a" * 40, split_fingerprint="b" * 64, config_sha256="c" * 64,
                      seed=0, student_input=[64, 64], teacher_input=[128, 128], data_use="validation", synthetic=True)
        return record

    def test_complete_record(self):
        self.assertTrue(validate_run_record(self.valid_record())["valid"])

    def test_collects_missing_reproducibility_fields(self):
        report = validate_run_record({})
        self.assertFalse(report["valid"])
        self.assertIn("code_commit must contain 40 hexadecimal characters", report["errors"])

    def test_rejects_wrong_types_and_unlabeled_results(self):
        record = self.valid_record()
        record.update(seed=True, synthetic="yes", student_input=[0, 32], data_use="test-tuning")
        self.assertEqual(len(validate_run_record(record)["errors"]), 4)
