import unittest
from pathlib import Path

from asd import ASDConfig
from asd.reproduce.dspark import DSparkASDConfig

ROOT = Path(__file__).resolve().parents[1]


class PublicSurfaceSeparationTests(unittest.TestCase):
    def test_qwen_api_and_config_remain_the_primary_surface(self) -> None:
        qwen = ASDConfig.from_json(ROOT / "configs" / "dspark_stable.json")
        self.assertEqual(qwen.risk_budget, 8.0)
        self.assertEqual(qwen.max_regret_per_value, 0.25)
        self.assertEqual(qwen.max_relaxed_tokens_per_block, 2)
        self.assertFalse(hasattr(qwen, "block_size"))

    def test_deepseek_config_is_an_isolated_experiment_type(self) -> None:
        deepseek = DSparkASDConfig.from_json(
            ROOT / "configs" / "deepseek_v4_flash_dspark_asd.json"
        )
        self.assertIsNot(ASDConfig, DSparkASDConfig)
        self.assertEqual(
            deepseek.to_mapping(),
            {
                "B": 2.0625,
                "g": 2.0625,
                "m": 1,
                "value_scheme": "normalized_suffix",
                "block_size": 5,
            },
        )


if __name__ == "__main__":
    unittest.main()
