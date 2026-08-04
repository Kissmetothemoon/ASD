import unittest

from asd.metrics import evaluate_pair


class MetricTests(unittest.TestCase):
    def test_valid_pair(self):
        result = evaluate_pair(
            baseline_pre={"completion_tokens_per_s": 100, "completion_tokens": 1000},
            candidate={
                "completion_tokens_per_s": 106,
                "completion_tokens": 1000,
                "delta_accuracy_pp": -0.5,
                "completion_length_delta_pct": 0.2,
            },
            baseline_post={"completion_tokens_per_s": 100, "completion_tokens": 1000},
        )
        self.assertAlmostEqual(result.delta_tps_pct, 6.0)
        self.assertTrue(result.protocol_valid)
        self.assertTrue(result.speed_target_met)
        self.assertTrue(result.quality_constraints_met)
        self.assertTrue(result.overall_success)

    def test_drift_failure_is_reported(self):
        result = evaluate_pair(
            baseline_pre={"tps": 100, "completion_tokens": 1000},
            candidate={"tps": 110, "completion_tokens": 1000},
            baseline_post={"tps": 110, "completion_tokens": 1000},
        )
        self.assertFalse(result.speed_eligible)
        self.assertIn("baseline_drift_exceeds_limit", result.reasons)

    def test_speed_eligibility_is_separate_from_target(self):
        result = evaluate_pair(
            baseline_pre={"tps": 100, "completion_tokens": 1000},
            candidate={"tps": 104, "completion_tokens": 1000},
            baseline_post={"tps": 100, "completion_tokens": 1000},
        )
        self.assertTrue(result.speed_eligible)
        self.assertFalse(result.speed_target_met)
        self.assertFalse(result.overall_success)

    def test_natural_eos_does_not_require_fixed_token_equality(self):
        result = evaluate_pair(
            baseline_pre={"tps": 100, "completion_tokens": 1000},
            candidate={
                "tps": 106,
                "completion_tokens": 980,
                "delta_accuracy_pp": -0.25,
                "completion_length_delta_pct": -2.0,
            },
            baseline_post={"tps": 100, "completion_tokens": 1004},
            fixed_workload=False,
        )
        self.assertTrue(result.protocol_valid)
        self.assertTrue(result.overall_success)


if __name__ == "__main__":
    unittest.main()
