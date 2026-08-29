"""Offline reduction of the frozen DSpark calibration experiment."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .asd_config import DSparkASDConfig
from .config import CALIBRATION_COUNT
from .io import load_jsonl, sha256_file, write_json, write_jsonl

_TRACE_FIELDS = {
    "snapshot_index",
    "proposal_ordinal",
    "request_serial",
    "rid",
    "request_pool_slot",
    "forward_ct",
    "barrier_position",
    "regret",
    "value",
    "regret_per_value",
}


def linear_q25(sorted_values: Sequence[float]) -> float:
    """Return the linearly interpolated 25th percentile used by the paper."""

    if not sorted_values:
        raise ValueError("positive first-rejection ratio distribution is empty")
    previous = -math.inf
    normalized: list[float] = []
    for position, value in enumerate(sorted_values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"q25 value {position} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0 or number < previous:
            raise ValueError(
                f"q25 value {position} must be sorted, positive, and finite"
            )
        normalized.append(number)
        previous = number
    h = (len(normalized) - 1) * 0.25
    lower = math.floor(h)
    upper = math.ceil(h)
    if lower == upper:
        return normalized[lower]
    return normalized[lower] * (upper - h) + normalized[upper] * (h - lower)


def _validate_outputs(path: Path, *, arm: str) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    if len(records) != CALIBRATION_COUNT:
        raise ValueError(
            f"{arm} outputs must contain exactly {CALIBRATION_COUNT} records"
        )
    for position, record in enumerate(records):
        token_ids = record.get("output_token_ids")
        if (
            record.get("cohort") != "calibration"
            or record.get("cohort_position") != position
            or record.get("terminal_status") != "succeeded"
            or not isinstance(record.get("dataset_index"), int)
            or not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in token_ids
            )
        ):
            raise ValueError(
                f"{arm} output {position} lacks its frozen identity or token IDs"
            )
    return records


def _response_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    response_ids: list[str] = []
    for position, record in enumerate(records):
        attempts = record.get("attempts")
        successful = (
            [
                attempt
                for attempt in attempts
                if isinstance(attempt, Mapping) and attempt.get("status") == "success"
            ]
            if isinstance(attempts, list)
            else []
        )
        response = successful[-1].get("raw_response") if len(successful) == 1 else None
        response_id = response.get("id") if isinstance(response, Mapping) else None
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"calibration output {position} has no response id")
        response_ids.append(response_id)
    if len(set(response_ids)) != len(response_ids):
        raise ValueError("calibration response ids must be unique")
    return response_ids


def compare_b0_outputs(
    native: Sequence[Mapping[str, Any]],
    b0: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Compare complete generated token-ID lists, never re-tokenized text."""

    mismatches: list[int] = []
    first: dict[str, Any] | None = None
    for position, (native_record, b0_record) in enumerate(zip(native, b0)):
        if native_record.get("dataset_index") != b0_record.get(
            "dataset_index"
        ) or native_record.get("cohort_position") != b0_record.get("cohort_position"):
            raise ValueError(f"native/B0 identity differs at position {position}")
        native_ids = list(native_record["output_token_ids"])
        b0_ids = list(b0_record["output_token_ids"])
        if native_ids == b0_ids:
            continue
        mismatches.append(position)
        if first is None:
            token_position = next(
                (
                    index
                    for index, pair in enumerate(zip(native_ids, b0_ids))
                    if pair[0] != pair[1]
                ),
                min(len(native_ids), len(b0_ids)),
            )
            first = {
                "schema_version": 1,
                "cohort_position": position,
                "dataset_index": native_record["dataset_index"],
                "first_differing_token_position": token_position,
                "native_output_token_ids": native_ids,
                "b0_output_token_ids": b0_ids,
            }
    return (
        {
            "schema_version": 1,
            "status": "PASS" if not mismatches else "FAIL",
            "comparison": "complete output_token_ids lists, no re-tokenization",
            "sample_count": len(native),
            "full_token_id_lists_equal": len(native) - len(mismatches),
            "mismatch_count": len(mismatches),
            "mismatch_cohort_positions": mismatches,
        },
        first,
    )


def _positive_trace_values(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        if set(record) != _TRACE_FIELDS:
            raise ValueError(f"trace row {position} has unexpected fields")
        integer_fields = (
            "snapshot_index",
            "proposal_ordinal",
            "request_serial",
            "request_pool_slot",
            "forward_ct",
            "barrier_position",
        )
        if any(
            isinstance(record[field], bool)
            or not isinstance(record[field], int)
            or record[field] < 0
            for field in integer_fields
        ):
            raise ValueError(f"trace row {position} has invalid integer fields")
        if record["request_serial"] <= 0 or not 0 <= record["barrier_position"] < 5:
            raise ValueError(f"trace row {position} has invalid DSpark position")
        if not isinstance(record["rid"], str) or not record["rid"]:
            raise ValueError(f"trace row {position} has no response id")
        regret, value, ratio = (
            record["regret"],
            record["value"],
            record["regret_per_value"],
        )
        if any(
            isinstance(number, bool) or not isinstance(number, (int, float))
            for number in (regret, value, ratio)
        ):
            raise ValueError(f"trace row {position} has non-numeric values")
        regret, value, ratio = float(regret), float(value), float(ratio)
        expected_value = (5 - record["barrier_position"]) / 5
        if (
            not all(math.isfinite(number) for number in (regret, value, ratio))
            or regret <= 0
            or value <= 0
            or ratio <= 0
            or not math.isclose(value, expected_value, rel_tol=0, abs_tol=1e-6)
            or not math.isclose(ratio, regret / value, rel_tol=1e-6, abs_tol=1e-7)
        ):
            raise ValueError(f"trace row {position} is not a valid positive ratio")
        values.append(
            {**record, "regret": regret, "value": value, "regret_per_value": ratio}
        )
    values.sort(
        key=lambda row: (
            row["regret_per_value"],
            row["snapshot_index"],
            row["proposal_ordinal"],
        )
    )
    return [
        {"sorted_ordinal": ordinal, **record} for ordinal, record in enumerate(values)
    ]


def reduce_calibration(
    *,
    native_outputs: Path,
    b0_outputs: Path,
    native_trace: Path,
    native_counters: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify B=0 equivalence and freeze the q25 ASD configuration."""

    native = _validate_outputs(native_outputs, arm="native")
    b0 = _validate_outputs(b0_outputs, arm="b0")
    equivalence, counterexample = compare_b0_outputs(native, b0)
    if equivalence["status"] != "PASS":
        raise ValueError(
            "B=0 is not token-identical to native DSpark; refusing to calibrate"
        )

    counters = json.loads(native_counters.read_text(encoding="utf-8"))
    trace = load_jsonl(native_trace)
    response_ids = _response_ids(native)
    trace_ids = {row.get("rid") for row in trace}
    if (
        not isinstance(counters, Mapping)
        or counters.get("status") != "PASS"
        or counters.get("arm") != "native-trace"
        or counters.get("trace_rows_dropped") != 0
        or counters.get("trace_rows_stored") != len(trace)
        or counters.get("native_acceptance_preserved") is not True
        or counters.get("cohort_response_ids") != response_ids
        or counters.get("trace_scope_proven") is not True
        or not trace_ids.issubset(set(response_ids))
    ):
        raise ValueError("native counters do not prove a complete scoped trace")

    values = _positive_trace_values(trace)
    q25 = linear_q25([row["regret_per_value"] for row in values])
    config = DSparkASDConfig(
        risk_budget=q25,
        max_regret_per_value=q25,
        max_relaxed_mismatches_per_block=1,
        value_scheme="normalized_suffix",
        block_size=5,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    equivalence.update(
        native_outputs_sha256=sha256_file(native_outputs),
        b0_outputs_sha256=sha256_file(b0_outputs),
    )
    write_json(output_dir / "b0_equivalence.json", equivalence, immutable=True)
    if counterexample is not None:
        write_json(
            output_dir / "first_minimal_counterexample.json",
            counterexample,
            immutable=True,
        )
    write_jsonl(output_dir / "calibration_values.jsonl", values, immutable=True)
    write_json(output_dir / "asd_config.json", config.to_mapping(), immutable=True)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "b0_status": "PASS",
        "positive_value_count": len(values),
        "q25_method": "linear interpolation with h=(n-1)*0.25",
        "q25": q25,
        "config": config.to_mapping(),
        "config_fingerprint": config.fingerprint(),
        "trace_rows_seen": counters.get("trace_rows_seen"),
        "trace_rows_stored": len(trace),
        "trace_rows_dropped": 0,
        "trace_scope_proven": True,
        "source_artifacts": {
            "native_outputs_sha256": sha256_file(native_outputs),
            "b0_outputs_sha256": sha256_file(b0_outputs),
            "native_trace_sha256": sha256_file(native_trace),
            "native_counters_sha256": sha256_file(native_counters),
        },
    }
    write_json(output_dir / "calibration_summary.json", summary, immutable=True)
    return summary
