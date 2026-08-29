"""Offline-only recomputation of request, answer, timing, and counter metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .answers import extract_model_answer, extract_reference_answer
from .config import PROPOSAL_WIDTH, RECORD_SCHEMA_VERSION, fingerprint_document
from .schema import validate_acceptance_counters, validate_asd_counters


def _require_nonnegative_int(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def recompute_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    timing: Mapping[str, Any] | None = None,
    expected_count: int | None = None,
    expected_cohort: str | None = None,
) -> dict[str, Any]:
    """Recompute all summary values without trusting a prior summary file."""

    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} records, got {len(records)}")
    counts = {
        "total_requests": len(records),
        "terminal_requests": 0,
        "success_requests": 0,
        "failed_requests": 0,
        "attempts_total": 0,
        "retry_attempts": 0,
        "requests_with_retry": 0,
        "matches": 0,
        "mismatches": 0,
        "parse_failures": 0,
        "completion_tokens": 0,
    }
    acceptance_available = 0
    proposal_count = 0
    proposed_tokens = 0
    accepted_tokens = 0
    accepted_by_position = [0] * PROPOSAL_WIDTH
    acceptance_length_sum = 0
    acceptance_length_count = 0
    asd_available = 0
    asd_integer_fields = (
        "runtime_calls",
        "relaxed_tokens",
        "budget_hit_requests",
        "cap_bound_blocks",
        "requests_initialized",
        "requests_finalized",
        "requests_aborted",
        "slot_resets",
        "state_leaks",
    )
    asd_totals: dict[str, int | float] = {field: 0 for field in asd_integer_fields}
    budget_spent_values: list[float] = []
    active_state_gauges: list[int] = []
    dataset_indices: list[int] = []

    for expected_position, record in enumerate(records):
        if record.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise ValueError("unsupported request record schema_version")
        if record.get("cohort_position") != expected_position:
            raise ValueError("records are not in cohort_position order")
        if expected_cohort is not None and record.get("cohort") != expected_cohort:
            raise ValueError(
                f"record {expected_position} is not cohort {expected_cohort!r}"
            )
        dataset_index = _require_nonnegative_int(record, "dataset_index")
        dataset_indices.append(dataset_index)
        terminal_status = record.get("terminal_status")
        if terminal_status not in {"succeeded", "failed"}:
            raise ValueError("record is not terminal")
        counts["terminal_requests"] += 1
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
            raise ValueError("attempts must contain between 1 and 3 entries")
        if record.get("attempt_count") != len(attempts):
            raise ValueError("attempt_count does not match attempts")
        retries = len(attempts) - 1
        if record.get("retry_count") != retries:
            raise ValueError("retry_count does not match attempts")
        counts["attempts_total"] += len(attempts)
        counts["retry_attempts"] += retries
        counts["requests_with_retry"] += int(retries > 0)
        if any(
            attempt.get("attempt_number") != number
            for number, attempt in enumerate(attempts, start=1)
        ):
            raise ValueError("attempt numbers are not contiguous")

        if terminal_status == "failed":
            if record.get("request_succeeded") is not False:
                raise ValueError("failed record claims request success")
            if any(attempt.get("status") == "success" for attempt in attempts):
                raise ValueError("failed record contains a successful attempt")
            if record.get("completion_tokens") != 0:
                raise ValueError("failed request must have zero completion tokens")
            counts["failed_requests"] += 1
            continue

        if record.get("request_succeeded") is not True:
            raise ValueError("succeeded record does not claim request success")
        if attempts[-1].get("status") != "success":
            raise ValueError("succeeded record's final attempt is not successful")
        if "raw_response" not in attempts[-1]:
            raise ValueError("successful attempt does not preserve raw response")
        token_ids = record.get("output_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("successful record has no output_token_ids")
        completion_tokens = _require_nonnegative_int(record, "completion_tokens")
        if completion_tokens != len(token_ids):
            raise ValueError("completion_tokens does not equal output_token_ids length")
        counts["completion_tokens"] += completion_tokens
        counts["success_requests"] += 1

        reference = extract_reference_answer(str(record.get("reference_response", "")))
        model = extract_model_answer(str(record.get("model_text", "")))
        if reference.value != record.get("reference_answer"):
            raise ValueError("stored reference answer does not re-extract")
        if model.value != record.get("extracted_answer"):
            raise ValueError("stored model answer does not re-extract")
        if reference.value is None or model.value is None:
            if record.get("parse_status") not in {
                "parse_failure",
                "reference_parse_failure",
            }:
                raise ValueError("parse failure status is inconsistent")
            if record.get("answer_match") is not None:
                raise ValueError("parse failure must not claim a match")
            counts["parse_failures"] += 1
        else:
            matched = model.value == reference.value
            if record.get("parse_status") != "parsed":
                raise ValueError("parsed answer has wrong parse_status")
            if record.get("answer_match") is not matched:
                raise ValueError("answer_match is inconsistent")
            counts["matches" if matched else "mismatches"] += 1

        acceptance = record.get("acceptance_counters")
        if acceptance is not None:
            normalized_acceptance = validate_acceptance_counters(acceptance)
            acceptance_available += 1
            proposal_count += normalized_acceptance["proposal_count"]
            proposed_tokens += normalized_acceptance["draft_tokens_proposed"]
            accepted_tokens += normalized_acceptance["accepted_draft_tokens"]
            for position, value in enumerate(
                normalized_acceptance["accepted_draft_tokens_by_position"]
            ):
                accepted_by_position[position] += value
            length_sum = normalized_acceptance["acceptance_length_including_bonus_sum"]
            length_count = normalized_acceptance[
                "acceptance_length_including_bonus_count"
            ]
            if length_sum is not None and length_count is not None:
                acceptance_length_sum += length_sum
                acceptance_length_count += length_count

        asd = record.get("asd_counters")
        if asd is not None:
            normalized_asd = validate_asd_counters(asd)
            asd_available += 1
            for field in asd_integer_fields:
                asd_totals[field] = int(asd_totals[field]) + int(normalized_asd[field])
            budget_spent_values.append(normalized_asd["budget_spent"])
            active_state_gauges.append(normalized_asd["active_request_states"])

    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError("dataset_index values are not unique within cohort")
    all_terminal = counts["terminal_requests"] == len(records)
    if counts["success_requests"] + counts["failed_requests"] != len(records):
        raise AssertionError("success/failed counts do not cover all records")
    timed_wall_seconds: float | None = None
    output_tps: float | None = None
    if timing is not None:
        if timing.get("clock") != "time.monotonic_ns":
            raise ValueError("formal timing does not use time.monotonic_ns")
        if timing.get("warmup_included") is not False:
            raise ValueError("formal timing includes warmup")
        if timing.get("formal_record_count") != len(records):
            raise ValueError("formal timing record count mismatch")
        timed = timing.get("timed_wall_seconds")
        if (
            isinstance(timed, bool)
            or not isinstance(timed, (int, float))
            or not math.isfinite(float(timed))
            or float(timed) <= 0
        ):
            raise ValueError("timed_wall_seconds must be finite and positive")
        start = timing.get("formal_start_monotonic_ns")
        end = timing.get("formal_end_monotonic_ns")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or end <= start
        ):
            raise ValueError("invalid formal monotonic boundaries")
        derived = (end - start) / 1_000_000_000
        if not math.isclose(derived, float(timed), rel_tol=0, abs_tol=1e-12):
            raise ValueError("timed_wall_seconds does not match boundaries")
        timed_wall_seconds = float(timed)
        output_tps = counts["completion_tokens"] / timed_wall_seconds

    summary: dict[str, Any] = {
        "schema_version": 1,
        **counts,
        "all_terminal": all_terminal,
        "timing": {
            "timed_wall_seconds": timed_wall_seconds,
            "end_to_end_output_tps": output_tps,
            "warmup_included": False if timing is not None else None,
        },
        "acceptance": {
            "records_with_counters": acceptance_available,
            "proposal_count": proposal_count,
            "draft_tokens_proposed": proposed_tokens,
            "accepted_draft_tokens": accepted_tokens,
            "mean_accepted_draft_tokens_per_proposal": (
                accepted_tokens / proposal_count if proposal_count else None
            ),
            "accepted_draft_tokens_by_position": accepted_by_position,
            "acceptance_length_including_bonus_sum": acceptance_length_sum,
            "acceptance_length_including_bonus_count": acceptance_length_count,
            "mean_acceptance_length_including_bonus": (
                acceptance_length_sum / acceptance_length_count
                if acceptance_length_count
                else None
            ),
        },
        "asd": {
            "records_with_counters": asd_available,
            **asd_totals,
            "budget_spent": math.fsum(budget_spent_values),
            "max_active_request_states_after_terminal": (
                max(active_state_gauges) if active_state_gauges else None
            ),
            "final_active_request_states": (
                active_state_gauges[-1] if active_state_gauges else None
            ),
        },
    }
    summary["fingerprint_sha256"] = fingerprint_document(summary)
    return summary
