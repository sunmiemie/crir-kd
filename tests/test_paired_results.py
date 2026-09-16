import unittest
from crir_kd.paired_results import paired_results


class PairedTests(unittest.TestCase):
    def records(self):
        return [{"method": method, "seed": seed, "protocol_fingerprint": "a" * 64, "synthetic": True, "metrics": {"toy": value}}
                for method, seed, value in [("base", 0, 2), ("base", 1, 5), ("candidate", 0, 4), ("candidate", 2, 10)]]

    def test_pairs_common_seeds_and_reports_exclusions(self):
        report = paired_results(self.records(), "base", "candidate", "toy")
        self.assertEqual(report["mean_delta"], 2)
        self.assertEqual(report["unpaired_seeds"], {"base": [1], "candidate": [2]})
        self.assertIsNone(report["sample_std_delta"])

    def test_incompatible_protocol_is_not_compared(self):
        records = self.records()
        records[0]["protocol_fingerprint"] = "b" * 64
        with self.assertRaises(ValueError):
            paired_results(records, "base", "candidate", "toy")
