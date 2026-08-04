"""Paired B-C-B metric computation and success criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SuccessCriteria


@dataclass(frozen=True)
class PairedResult:
    baseline_tps: float
    candidate_tps: float
    delta_tps_pct: float
    baseline_drift_pct: float
    accuracy_delta_pp: float | None
    completion_length_delta_pct: float | None
    fixed_token_equal: bool | None
    fixed_workload: bool
    protocol_valid: bool
    speed_eligible: bool
    speed_target_met: bool
    quality_constraints_met: bool | None
    overall_success: bool
    reasons: tuple[str, ...]


def _value(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        raw = row.get(name)
        if raw not in (None, ""):
            return float(raw)
    return None


def evaluate_pair(
    *,
    baseline_pre: dict[str, Any],
    candidate: dict[str, Any],
    baseline_post: dict[str, Any],
    criteria: SuccessCriteria = SuccessCriteria(),
    fixed_workload: bool = True,
) -> PairedResult:
    pre_tps = _value(baseline_pre, "tps", "completion_tokens_per_s", "candidate_tps")
    candidate_tps = _value(candidate, "tps", "completion_tokens_per_s", "candidate_tps")
    post_tps = _value(baseline_post, "tps", "completion_tokens_per_s", "candidate_tps")
    if pre_tps is None or candidate_tps is None or post_tps is None:
        raise ValueError("all three rows must contain a throughput field")
    if pre_tps <= 0 or post_tps <= 0 or candidate_tps <= 0:
        raise ValueError("throughput values must be positive")

    baseline_tps = (pre_tps + post_tps) / 2.0
    delta_tps_pct = 100.0 * (candidate_tps / baseline_tps - 1.0)
    baseline_drift_pct = 100.0 * (post_tps / pre_tps - 1.0)

    pre_tokens = _value(baseline_pre, "completion_tokens", "baseline_completion_tokens")
    candidate_tokens = _value(candidate, "completion_tokens", "candidate_completion_tokens")
    post_tokens = _value(baseline_post, "completion_tokens", "baseline_completion_tokens")
    fixed_token_equal: bool | None = None
    if pre_tokens is not None and candidate_tokens is not None and post_tokens is not None:
        fixed_token_equal = (
            pre_tokens == candidate_tokens == post_tokens
        )

    accuracy_delta = _value(candidate, "accuracy_delta_pp", "delta_accuracy_pp")
    length_delta = _value(
        candidate,
        "completion_length_delta_pct",
        "completion_length_delta",
    )
    reasons: list[str] = []
    if abs(baseline_drift_pct) > criteria.maximum_baseline_drift_pct:
        reasons.append("baseline_drift_exceeds_limit")
    if (
        fixed_workload
        and criteria.require_fixed_token_equality_for_speed
        and fixed_token_equal is not True
    ):
        reasons.append("fixed_token_equality_missing_or_false")
    protocol_valid = not reasons
    speed_eligible = protocol_valid
    speed_target_met = (
        speed_eligible and delta_tps_pct >= criteria.minimum_tps_gain_pct
    )
    if speed_eligible and not speed_target_met:
        reasons.append("speed_target_not_met")

    quality_constraints_met: bool | None = None
    if accuracy_delta is not None and length_delta is not None:
        quality_constraints_met = True
        if accuracy_delta < -criteria.maximum_accuracy_drop_pp:
            reasons.append("accuracy_drop_exceeds_limit")
            quality_constraints_met = False
        if abs(length_delta) > criteria.maximum_completion_length_change_pct:
            reasons.append("completion_length_change_exceeds_limit")
            quality_constraints_met = False
    else:
        reasons.append("quality_metrics_not_evaluated")

    overall_success = speed_target_met and quality_constraints_met is True

    return PairedResult(
        baseline_tps=baseline_tps,
        candidate_tps=candidate_tps,
        delta_tps_pct=delta_tps_pct,
        baseline_drift_pct=baseline_drift_pct,
        accuracy_delta_pp=accuracy_delta,
        completion_length_delta_pct=length_delta,
        fixed_token_equal=fixed_token_equal,
        fixed_workload=fixed_workload,
        protocol_valid=protocol_valid,
        speed_eligible=speed_eligible,
        speed_target_met=speed_target_met,
        quality_constraints_met=quality_constraints_met,
        overall_success=overall_success,
        reasons=tuple(dict.fromkeys(reasons)),
    )
