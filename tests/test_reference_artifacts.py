import json
import unittest
from pathlib import Path

from asd.reproduce.dspark.asd_config import DSparkASDConfig
from asd.reproduce.dspark.config import (
    DATASET_REVISION,
    MODEL_REPO_ID,
    MODEL_REVISION,
    SGLANG_COMMIT,
    SHUFFLE_SEED,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "deepseek-v4-flash-dspark"


class ReferenceArtifactTests(unittest.TestCase):
    def test_protocol_matches_code_pins(self):
        protocol = json.loads((EXPERIMENT / "protocol.json").read_text())
        self.assertEqual(protocol["model"]["repo_id"], MODEL_REPO_ID)
        self.assertEqual(protocol["model"]["huggingface_revision"], MODEL_REVISION)
        self.assertEqual(protocol["dataset"]["revision"], DATASET_REVISION)
        self.assertEqual(protocol["dataset"]["seed"], SHUFFLE_SEED)
        self.assertEqual(protocol["runtime"]["sglang_commit"], SGLANG_COMMIT)

    def test_reference_config_is_the_committed_config(self):
        results = json.loads((EXPERIMENT / "reference_results.json").read_text())
        committed = DSparkASDConfig.from_json(
            ROOT / "configs" / "deepseek_v4_flash_dspark_asd.json"
        )
        self.assertEqual(results["method_identity"]["config"], committed.to_mapping())

    def test_reference_tps_delta_recomputes(self):
        results = json.loads((EXPERIMENT / "reference_results.json").read_text())
        native = results["formal"]["native"]["end_to_end_output_tps"]
        asd = results["formal"]["asd"]["end_to_end_output_tps"]
        expected = (asd / native - 1) * 100
        self.assertAlmostEqual(
            expected, results["delta"]["end_to_end_output_tps_percent"]
        )

    def test_public_reference_contains_no_internal_machine_identity(self):
        text = (EXPERIMENT / "reference_results.json").read_text()
        for forbidden in ("/mnt/", "/home/", "/mlx_devbox/", "worker_id", "4106666"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
