"""Tests for the minimal ASD decode identity."""

from __future__ import annotations

import hashlib
import json
import math
import unittest

from asd.reproduce.dspark.asd_config import NORMALIZED_SUFFIX, DSparkASDConfig


def make_config(
    *,
    B: float = 0.25,
    g: float = 0.25,
    m: int = 1,
    value_scheme: str = NORMALIZED_SUFFIX,
    block_size: int = 5,
) -> DSparkASDConfig:
    return DSparkASDConfig(
        risk_budget=B,
        max_regret_per_value=g,
        max_relaxed_mismatches_per_block=m,
        value_scheme=value_scheme,
        block_size=block_size,
    )


def hash_mapping(mapping: dict[str, float | int | str]) -> str:
    payload = json.dumps(
        mapping,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class ConfigTests(unittest.TestCase):
    def test_canonical_B_g_m_mapping_round_trips(self) -> None:
        original = make_config()
        mapping = {
            "B": 0.25,
            "g": 0.25,
            "m": 1,
            "value_scheme": "normalized_suffix",
            "block_size": 5,
        }
        self.assertEqual(original.to_mapping(), mapping)
        self.assertEqual(DSparkASDConfig.from_mapping(mapping), original)
        self.assertEqual(original.B, original.risk_budget)
        self.assertEqual(original.g, original.max_regret_per_value)
        self.assertEqual(
            original.m,
            original.max_relaxed_mismatches_per_block,
        )

    def test_unknown_reporting_or_success_keys_fail_loudly(self) -> None:
        mapping = make_config().to_mapping()
        mapping["success_criteria"] = {"minimum_tps_gain_pct": 5}
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            DSparkASDConfig.from_mapping(mapping)

    def test_only_normalized_suffix_is_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized_suffix"):
            make_config(value_scheme="suffix")

    def test_sampling_and_reporting_fields_are_not_in_decode_mapping(self) -> None:
        self.assertEqual(
            set(make_config().to_mapping()),
            {"B", "g", "m", "value_scheme", "block_size"},
        )

    def test_fingerprint_is_stable(self) -> None:
        first = make_config()
        second = make_config()
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(first.fingerprint(), hash_mapping(first.to_mapping()))

    def test_fingerprint_changes_with_B_g_m_and_block_size(self) -> None:
        baseline = make_config().fingerprint()
        variants = (
            make_config(B=0.5),
            make_config(g=0.5),
            make_config(m=2),
            make_config(block_size=6),
        )
        for variant in variants:
            with self.subTest(variant=variant.to_mapping()):
                self.assertNotEqual(baseline, variant.fingerprint())

    def test_fingerprint_payload_covers_value_scheme(self) -> None:
        # The public config rejects non-contract schemes.  Compare the
        # canonical payload with a hypothetical changed schema value to prove
        # that the fingerprint itself includes this field.
        baseline = make_config()
        hypothetical = baseline.to_mapping()
        hypothetical["value_scheme"] = "hypothetical_other_scheme"
        self.assertNotEqual(
            baseline.fingerprint(),
            hash_mapping(hypothetical),
        )

    def test_block_size_validation_fails_loudly(self) -> None:
        cfg = make_config()
        cfg.validate_block_size(5)
        with self.assertRaisesRegex(ValueError, "observed block_size=4"):
            cfg.validate_block_size(4)

    def test_invalid_numeric_values_fail_loudly(self) -> None:
        for field, value in (
            ("B", -0.1),
            ("B", math.inf),
            ("g", -0.1),
            ("g", math.nan),
        ):
            kwargs = {field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                make_config(**kwargs)
        with self.assertRaises(ValueError):
            make_config(m=-1)
        with self.assertRaises(ValueError):
            make_config(block_size=0)


if __name__ == "__main__":
    unittest.main()
