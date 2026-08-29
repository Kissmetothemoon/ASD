"""Minimal, engine-neutral configuration for the ASD decision rule."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

NORMALIZED_SUFFIX: Final = "normalized_suffix"

_MAPPING_KEYS: Final = frozenset({"B", "g", "m", "value_scheme", "block_size"})


def _finite_non_negative(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    # Keep semantically identical +0.0 and -0.0 fingerprints identical.
    return 0.0 if normalized == 0.0 else normalized


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class DSparkASDConfig:
    """All and only the fields that affect an ASD prefix decision.

    The long Python field names keep the interface readable.  The serialized
    mapping uses the experiment-contract names ``B``, ``g`` and ``m``:

    - ``risk_budget`` / ``B`` is the request-wide regret budget.
    - ``max_regret_per_value`` / ``g`` is the per-mismatch ratio gate.
    - ``max_relaxed_mismatches_per_block`` / ``m`` is the per-block cap.

    This pure core intentionally implements only ``normalized_suffix``.  The
    source repository's alternative value schemes are outside the frozen V4
    experiment contract.
    """

    risk_budget: float
    max_regret_per_value: float
    max_relaxed_mismatches_per_block: int
    block_size: int
    value_scheme: str = NORMALIZED_SUFFIX

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "risk_budget",
            _finite_non_negative(self.risk_budget, name="risk_budget (B)"),
        )
        object.__setattr__(
            self,
            "max_regret_per_value",
            _finite_non_negative(
                self.max_regret_per_value,
                name="max_regret_per_value (g)",
            ),
        )
        object.__setattr__(
            self,
            "max_relaxed_mismatches_per_block",
            _non_negative_integer(
                self.max_relaxed_mismatches_per_block,
                name="max_relaxed_mismatches_per_block (m)",
            ),
        )
        block_size = _non_negative_integer(self.block_size, name="block_size")
        if block_size == 0:
            raise ValueError("block_size must be positive")
        object.__setattr__(self, "block_size", block_size)
        if self.value_scheme != NORMALIZED_SUFFIX:
            raise ValueError(
                "value_scheme must be exactly "
                f"{NORMALIZED_SUFFIX!r}, got {self.value_scheme!r}"
            )

    @property
    def B(self) -> float:
        """Canonical experiment name for the request-wide risk budget."""

        return self.risk_budget

    @property
    def g(self) -> float:
        """Canonical experiment name for the regret/value gate."""

        return self.max_regret_per_value

    @property
    def m(self) -> int:
        """Canonical experiment name for the per-block mismatch cap."""

        return self.max_relaxed_mismatches_per_block

    @property
    def max_relaxed_tokens_per_block(self) -> int:
        """Backward-compatible public alias for the mismatch cap."""

        return self.max_relaxed_mismatches_per_block

    def to_mapping(self) -> dict[str, float | int | str]:
        """Return the canonical decode-affecting mapping."""

        return {
            "B": self.B,
            "g": self.g,
            "m": self.m,
            "value_scheme": self.value_scheme,
            "block_size": self.block_size,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DSparkASDConfig:
        """Build a config from the strict canonical mapping.

        Unknown and missing fields fail loudly.  Reporting metadata and legacy
        success criteria therefore cannot silently enter the decode identity.
        """

        keys = frozenset(value)
        unknown = sorted(keys - _MAPPING_KEYS)
        missing = sorted(_MAPPING_KEYS - keys)
        if unknown or missing:
            details = []
            if unknown:
                details.append("unknown keys: " + ", ".join(unknown))
            if missing:
                details.append("missing keys: " + ", ".join(missing))
            raise ValueError("; ".join(details))
        return cls(
            risk_budget=value["B"],
            max_regret_per_value=value["g"],
            max_relaxed_mismatches_per_block=value["m"],
            value_scheme=value["value_scheme"],
            block_size=value["block_size"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DSparkASDConfig:
        """Load the canonical decode-affecting mapping from JSON."""

        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("configuration must be a JSON object")
        return cls.from_mapping(value)

    def fingerprint(self) -> str:
        """Return a stable SHA-256 over every decode-affecting field."""

        payload = json.dumps(
            self.to_mapping(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def validate_block_size(self, observed: int) -> None:
        """Fail loudly if the running drafter width differs from the config."""

        observed_block_size = _non_negative_integer(
            observed,
            name="observed block_size",
        )
        if observed_block_size != self.block_size:
            raise ValueError(
                f"config block_size={self.block_size} but observed "
                f"block_size={observed_block_size}"
            )
