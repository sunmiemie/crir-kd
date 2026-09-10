import unittest
from test_protocols import protocol_fixture
from crir_kd.experiment_matrix import experiment_matrix


class MatrixTests(unittest.TestCase):
    def test_matrix_is_unique_and_budget_difference_explicit(self):
        result = experiment_matrix(protocol_fixture(), [{"name": "quality", "teacher_forwards_per_step": 1}, {"name": "joint", "teacher_forwards_per_step": 3}], [0, 1, 2])
        self.assertEqual(result["planned_run_count"], 6)
        self.assertEqual(len({run["run_id"] for run in result["runs"]}), 6)
        self.assertFalse(result["compute_matched"])
        self.assertTrue(all(run["status"] == "planned-not-executed" for run in result["runs"]))

    def test_joint_requires_quality_control_and_distinct_seeds(self):
        with self.assertRaises(ValueError):
            experiment_matrix(protocol_fixture(), [{"name": "joint", "teacher_forwards_per_step": 3}], [0])
        with self.assertRaises(ValueError):
            experiment_matrix(protocol_fixture(), [{"name": "quality", "teacher_forwards_per_step": 1}], [0, 0])
