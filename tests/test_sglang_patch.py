from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "sglang-dspark"
PATCH = INTEGRATION / "sglang-0.5.16-asd.patch"


class SGLangPatchTests(unittest.TestCase):
    def test_manifest_matches_patch(self) -> None:
        manifest = json.loads((INTEGRATION / "manifest.json").read_text())
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["patch"]["sha256"])
        self.assertEqual(manifest["sglang"]["version"], "0.5.16")
        self.assertEqual(
            manifest["sglang"]["base_commit"],
            "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1",
        )
        self.assertEqual(
            hashlib.sha256(
                (INTEGRATION / "runtime" / "pyproject.toml").read_bytes()
            ).hexdigest(),
            manifest["runtime"]["pyproject_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(
                (INTEGRATION / "runtime" / "uv.lock").read_bytes()
            ).hexdigest(),
            manifest["runtime"]["uv_lock_sha256"],
        )

    def test_patch_has_only_the_public_method_name(self) -> None:
        text = PATCH.read_text(encoding="utf-8")
        self.assertNotIn("HEDGE", text)
        self.assertNotIn("Hedge", text)
        self.assertNotIn("hedge", text)
        self.assertIn(
            "from asd.reproduce.dspark.asd_config import DSparkASDConfig", text
        )
        self.assertIn(
            "from asd.reproduce.dspark.torch_rule import choose_prefix_batch",
            text,
        )

    def test_patch_contains_no_internal_paths_or_credentials(self) -> None:
        text = PATCH.read_text(encoding="utf-8")
        for forbidden in (
            "/mnt/",
            "/home/",
            "/mlx_devbox/",
            "HF_TOKEN",
            "API_KEY",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_expected_sglang_files_are_patched(self) -> None:
        paths = {
            line.split(" b/", 1)[1]
            for line in PATCH.read_text(encoding="utf-8").splitlines()
            if line.startswith("diff --git a/")
        }
        self.assertEqual(
            paths,
            {
                "python/sglang/srt/entrypoints/openai/serving_chat.py",
                "python/sglang/srt/managers/scheduler_components/batch_result_processor.py",
                "python/sglang/srt/speculative/dspark_components/dspark_verify.py",
                "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py",
                "python/sglang/srt/speculative/dspark_components/asd_dspark.py",
            },
        )


if __name__ == "__main__":
    unittest.main()
