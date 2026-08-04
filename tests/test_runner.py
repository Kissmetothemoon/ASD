import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_deepspec_bcb import main


class RunnerTests(unittest.TestCase):
    def test_triplet_uses_strict_baselines_and_applies_gpu_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "runner.py"
            runner.write_text("# test placeholder\n", encoding="utf-8")
            output = root / "output"
            argv = [
                "run_deepspec_bcb.py",
                "--runner",
                str(runner),
                "--target",
                "target",
                "--draft",
                "draft",
                "--dataset-root",
                str(root),
                "--output-root",
                str(output),
                "--gpu",
                "3",
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.run_deepspec_bcb.subprocess.run"
            ) as run:
                self.assertEqual(main(), 0)

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                [item["risk_budget"] for item in manifest["runs"]],
                [0.0, 8.0, 0.0],
            )
            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")


if __name__ == "__main__":
    unittest.main()
