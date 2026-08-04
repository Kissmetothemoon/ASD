import json
import tempfile
import unittest
from pathlib import Path

from asd.config import ASDConfig


class ConfigTests(unittest.TestCase):
    def test_load_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "risk_budget": 8,
                        "max_regret_per_value": 0.25,
                        "max_relaxed_tokens_per_block": 2,
                    }
                ),
                encoding="utf-8",
            )
            config = ASDConfig.from_json(path)
        self.assertEqual(config.risk_budget, 8.0)

    def test_sampling_is_rejected(self):
        with self.assertRaises(ValueError):
            ASDConfig(8.0, 0.25, 2, temperature=0.7)

    def test_non_finite_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            ASDConfig(float("nan"), 0.25, 2)

    def test_method_name_is_fixed(self):
        with self.assertRaises(ValueError):
            ASDConfig(8.0, 0.25, 2, method_name="OTHER")

    def test_fingerprint_is_stable_and_configuration_sensitive(self):
        first = ASDConfig(8.0, 0.25, 2)
        second = ASDConfig(8.0, 0.25, 2)
        changed = ASDConfig(9.0, 0.25, 2)
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())


if __name__ == "__main__":
    unittest.main()
