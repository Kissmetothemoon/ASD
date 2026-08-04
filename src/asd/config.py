"""Configuration objects and validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SuccessCriteria:
    minimum_tps_gain_pct: float = 5.0
    maximum_baseline_drift_pct: float = 3.0
    maximum_accuracy_drop_pp: float = 1.0
    maximum_completion_length_change_pct: float = 3.0
    require_fixed_token_equality_for_speed: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_tps_gain_pct):
            raise ValueError("minimum_tps_gain_pct must be finite")
        if (
            not math.isfinite(self.maximum_baseline_drift_pct)
            or self.maximum_baseline_drift_pct < 0
        ):
            raise ValueError("maximum_baseline_drift_pct must be non-negative")
        if (
            not math.isfinite(self.maximum_accuracy_drop_pp)
            or self.maximum_accuracy_drop_pp < 0
        ):
            raise ValueError("maximum_accuracy_drop_pp must be non-negative")
        if (
            not math.isfinite(self.maximum_completion_length_change_pct)
            or self.maximum_completion_length_change_pct < 0
        ):
            raise ValueError(
                "maximum_completion_length_change_pct must be non-negative"
            )


@dataclass(frozen=True)
class ASDConfig:
    risk_budget: float
    max_regret_per_value: float
    max_relaxed_tokens_per_block: int
    temperature: float = 0.0
    method_name: str = "ASD"
    evidence_status: str = "candidate"
    success_criteria: SuccessCriteria = SuccessCriteria()

    def __post_init__(self) -> None:
        if not math.isfinite(self.risk_budget) or self.risk_budget < 0:
            raise ValueError("risk_budget must be non-negative")
        if (
            not math.isfinite(self.max_regret_per_value)
            or self.max_regret_per_value < 0
        ):
            raise ValueError("max_regret_per_value must be non-negative")
        if self.max_relaxed_tokens_per_block < 0:
            raise ValueError("max_relaxed_tokens_per_block must be non-negative")
        if self.temperature != 0.0:
            raise ValueError("the ASD verifier requires temperature=0")
        if self.method_name != "ASD":
            raise ValueError("method_name must be ASD")
        if self.evidence_status not in {
            "candidate",
            "frozen_candidate",
            "discovery_only",
        }:
            raise ValueError("unsupported evidence_status")

    def fingerprint(self) -> str:
        """Return a stable hash for distributed rank-consistency checks."""
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ASDConfig":
        criteria = SuccessCriteria(**value.get("success_criteria", {}))
        return cls(
            risk_budget=float(value["risk_budget"]),
            max_regret_per_value=float(value["max_regret_per_value"]),
            max_relaxed_tokens_per_block=int(
                value["max_relaxed_tokens_per_block"]
            ),
            temperature=float(value.get("temperature", 0.0)),
            method_name=str(value.get("method_name", "ASD")),
            evidence_status=str(value.get("evidence_status", "candidate")),
            success_criteria=criteria,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ASDConfig":
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("configuration must be a JSON object")
        return cls.from_mapping(value)
