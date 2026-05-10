import unittest

from crir_kd.summaries import summarize_runs


class SummaryTests(unittest.TestCase):
    def run_record(self, seed, value, protocol="a" * 64):
        return {"method": "q-only", "protocol_fingerprint": protocol, "seed": seed,
                "synthetic": True, "metrics": {"toy_score": value}}

    def test_mean_sample_std_and_single_seed(self):
        report = summarize_runs([self.run_record(0, 1), self.run_record(1, 3)])
        metric = report["groups"][0]["metrics"]["toy_score"]
        self.assertEqual(metric["mean"], 2)
        self.assertAlmostEqual(metric["sample_std"], 2 ** .5)
        single = summarize_runs([self.run_record(0, 1)])["groups"][0]
        self.assertTrue(single["single_run"])
        self.assertIsNone(single["metrics"]["toy_score"]["sample_std"])

    def test_different_protocols_are_never_pooled(self):
        report = summarize_runs([self.run_record(0, 1), self.run_record(0, 3, "b" * 64)])
        self.assertEqual(len(report["groups"]), 2)

    def test_duplicate_seed_missing_metric_and_nan_fail(self):
        with self.assertRaises(ValueError):
            summarize_runs([self.run_record(0, 1), self.run_record(0, 3)])
        with self.assertRaises(ValueError):
            summarize_runs([self.run_record(0, float("nan"))])
        bad = self.run_record(1, 2)
        bad["metrics"] = {"different": 1}
        with self.assertRaises(ValueError):
            summarize_runs([self.run_record(0, 1), bad])
