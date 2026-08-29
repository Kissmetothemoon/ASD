"""Validation for response metadata and speculative-decoding counters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import (
    COUNTER_SCHEMA_VERSION,
    PROPOSAL_WIDTH,
)


class ResponseSchemaError(ValueError):
    """The server response cannot support an auditable protocol record."""


@dataclass(frozen=True)
class ParsedResponse:
    model_text: str
    output_token_ids: list[int]
    completion_tokens: int
    acceptance_counters: dict[str, Any] | None
    asd_counters: dict[str, Any] | None


_SGLANG_SPEC_FIELDS = {
    "spec_verify_ct",
    "spec_num_proposed_drafts",
    "spec_proposed_drafts",
    "spec_num_correct_drafts",
    "spec_accepted_drafts",
    "spec_correct_drafts_histogram",
    "spec_accept_histogram",
    "spec_accept_rate",
    "spec_accept_length",
    "completion_tokens",
}


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponseSchemaError(f"{field} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field=field)


def validate_acceptance_counters(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponseSchemaError("acceptance_counters must be an object")
    required = {
        "schema_version",
        "proposal_count",
        "draft_tokens_proposed",
        "accepted_draft_tokens",
        "accepted_draft_tokens_by_position",
        "acceptance_length_including_bonus_sum",
        "acceptance_length_including_bonus_count",
    }
    if set(value) != required:
        raise ResponseSchemaError(
            "acceptance_counters fields mismatch: "
            f"expected {sorted(required)}, got {sorted(value)}"
        )
    if value["schema_version"] != COUNTER_SCHEMA_VERSION:
        raise ResponseSchemaError("unsupported acceptance counter schema_version")
    proposal_count = _nonnegative_int(value["proposal_count"], field="proposal_count")
    proposed = _nonnegative_int(
        value["draft_tokens_proposed"], field="draft_tokens_proposed"
    )
    accepted = _nonnegative_int(
        value["accepted_draft_tokens"], field="accepted_draft_tokens"
    )
    by_position = value["accepted_draft_tokens_by_position"]
    if not isinstance(by_position, list) or len(by_position) != PROPOSAL_WIDTH:
        raise ResponseSchemaError(
            "accepted_draft_tokens_by_position must have proposal width "
            f"{PROPOSAL_WIDTH}"
        )
    normalized_positions = [
        _nonnegative_int(item, field=f"accepted_draft_tokens_by_position[{index}]")
        for index, item in enumerate(by_position)
    ]
    if proposed > proposal_count * PROPOSAL_WIDTH:
        raise ResponseSchemaError(
            "draft_tokens_proposed exceeds proposal_count * proposal width"
        )
    if accepted > proposed:
        raise ResponseSchemaError("accepted_draft_tokens exceeds draft_tokens_proposed")
    if sum(normalized_positions) != accepted:
        raise ResponseSchemaError(
            "accepted_draft_tokens_by_position sum does not match accepted total"
        )
    if any(item > proposal_count for item in normalized_positions):
        raise ResponseSchemaError("accepted count at a position exceeds proposal_count")
    length_sum = _optional_nonnegative_int(
        value["acceptance_length_including_bonus_sum"],
        field="acceptance_length_including_bonus_sum",
    )
    length_count = _optional_nonnegative_int(
        value["acceptance_length_including_bonus_count"],
        field="acceptance_length_including_bonus_count",
    )
    if (length_sum is None) != (length_count is None):
        raise ResponseSchemaError(
            "acceptance length sum/count must both be null or both be integers"
        )
    return {
        "schema_version": COUNTER_SCHEMA_VERSION,
        "proposal_count": proposal_count,
        "draft_tokens_proposed": proposed,
        "accepted_draft_tokens": accepted,
        "accepted_draft_tokens_by_position": normalized_positions,
        "acceptance_length_including_bonus_sum": length_sum,
        "acceptance_length_including_bonus_count": length_count,
    }


def validate_asd_counters(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponseSchemaError("asd_counters must be an object")
    integer_fields = (
        "runtime_calls",
        "relaxed_tokens",
        "budget_hit_requests",
        "cap_bound_blocks",
        "requests_initialized",
        "requests_finalized",
        "requests_aborted",
        "slot_resets",
        "active_request_states",
        "state_leaks",
    )
    required = {"schema_version", "budget_spent", *integer_fields}
    if set(value) != required:
        raise ResponseSchemaError(
            "asd_counters fields mismatch: "
            f"expected {sorted(required)}, got {sorted(value)}"
        )
    if value["schema_version"] != COUNTER_SCHEMA_VERSION:
        raise ResponseSchemaError("unsupported ASD counter schema_version")
    normalized: dict[str, Any] = {"schema_version": COUNTER_SCHEMA_VERSION}
    for field in integer_fields:
        normalized[field] = _nonnegative_int(value[field], field=field)
    if normalized["budget_hit_requests"] not in (0, 1):
        raise ResponseSchemaError("budget_hit_requests must be 0 or 1")
    spent = value["budget_spent"]
    if isinstance(spent, bool) or not isinstance(spent, (int, float)):
        raise ResponseSchemaError("budget_spent must be a finite number")
    spent_float = float(spent)
    if not math.isfinite(spent_float) or spent_float < 0:
        raise ResponseSchemaError("budget_spent must be finite and non-negative")
    normalized["budget_spent"] = spent_float
    return normalized


def normalize_sglang_spec_acceptance(
    meta_info: Mapping[str, Any], *, completion_tokens: int
) -> dict[str, Any] | None:
    """Normalize SGLang's per-response ``spec_*`` fields and cross-check aliases."""

    present = _SGLANG_SPEC_FIELDS & set(meta_info)
    if not present:
        return None
    if present != _SGLANG_SPEC_FIELDS:
        missing = sorted(_SGLANG_SPEC_FIELDS - present)
        raise ResponseSchemaError(
            "SGLang speculative metadata is incomplete: " + ", ".join(missing)
        )
    proposals = _nonnegative_int(meta_info["spec_verify_ct"], field="spec_verify_ct")
    proposed = _nonnegative_int(
        meta_info["spec_num_proposed_drafts"],
        field="spec_num_proposed_drafts",
    )
    proposed_alias = _nonnegative_int(
        meta_info["spec_proposed_drafts"], field="spec_proposed_drafts"
    )
    accepted = _nonnegative_int(
        meta_info["spec_num_correct_drafts"],
        field="spec_num_correct_drafts",
    )
    accepted_alias = _nonnegative_int(
        meta_info["spec_accepted_drafts"], field="spec_accepted_drafts"
    )
    histogram = meta_info["spec_correct_drafts_histogram"]
    histogram_alias = meta_info["spec_accept_histogram"]
    if not isinstance(histogram, list) or len(histogram) != PROPOSAL_WIDTH + 1:
        raise ResponseSchemaError(
            "spec_correct_drafts_histogram must contain lengths 0 through 5"
        )
    normalized_histogram = [
        _nonnegative_int(value, field=f"spec_correct_drafts_histogram[{index}]")
        for index, value in enumerate(histogram)
    ]
    if not isinstance(histogram_alias, list):
        raise ResponseSchemaError("spec_accept_histogram must be an array")
    normalized_alias = [
        _nonnegative_int(value, field=f"spec_accept_histogram[{index}]")
        for index, value in enumerate(histogram_alias)
    ]
    metadata_completion = _nonnegative_int(
        meta_info["completion_tokens"], field="meta_info.completion_tokens"
    )
    derived_proposed = proposals * PROPOSAL_WIDTH
    derived_accepted = sum(
        accepted_length * count
        for accepted_length, count in enumerate(normalized_histogram)
    )
    by_position = [
        sum(normalized_histogram[position + 1 :]) for position in range(PROPOSAL_WIDTH)
    ]
    rate = meta_info["spec_accept_rate"]
    length = meta_info["spec_accept_length"]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (rate, length)
    ):
        raise ResponseSchemaError("spec_accept_rate/length must be numeric")
    expected_rate = derived_accepted / derived_proposed if derived_proposed else 0.0
    expected_length = metadata_completion / proposals if proposals else 0.0
    checks = {
        "histogram_count": sum(normalized_histogram) == proposals,
        "proposed": proposed == derived_proposed == proposed_alias,
        "accepted": accepted == derived_accepted == accepted_alias,
        "histogram_alias": normalized_alias == normalized_histogram,
        "completion_tokens": metadata_completion == completion_tokens,
        "accept_rate": math.isclose(
            float(rate), expected_rate, rel_tol=1e-12, abs_tol=1e-12
        ),
        "accept_length": math.isclose(
            float(length), expected_length, rel_tol=1e-12, abs_tol=1e-12
        ),
        "accepted_within_completion": derived_accepted <= completion_tokens,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ResponseSchemaError(
            "SGLang speculative metadata cross-check failed: " + ", ".join(failed)
        )
    return validate_acceptance_counters(
        {
            "schema_version": COUNTER_SCHEMA_VERSION,
            "proposal_count": proposals,
            "draft_tokens_proposed": proposed,
            "accepted_draft_tokens": accepted,
            "accepted_draft_tokens_by_position": by_position,
            "acceptance_length_including_bonus_sum": completion_tokens,
            "acceptance_length_including_bonus_count": proposals,
        }
    )


def parse_chat_response(response: Any) -> ParsedResponse:
    """Validate and extract the canonical OpenAI chat response seam."""

    if not isinstance(response, Mapping):
        raise ResponseSchemaError("response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ResponseSchemaError("response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ResponseSchemaError("choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ResponseSchemaError("choice.message must be an object")
    model_text = message.get("content")
    if not isinstance(model_text, str) or not model_text:
        raise ResponseSchemaError("choice.message.content must be non-empty")
    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, Mapping):
        raise ResponseSchemaError(
            "choices[0].meta_info is required for output token IDs"
        )
    token_ids = meta_info.get("output_token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ResponseSchemaError(
            "choices[0].meta_info.output_token_ids must be a non-empty array"
        )
    normalized_ids: list[int] = []
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ResponseSchemaError(
                f"output_token_ids[{index}] must be a non-negative integer"
            )
        normalized_ids.append(token_id)
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise ResponseSchemaError("response.usage must be an object")
    completion_tokens = _nonnegative_int(
        usage.get("completion_tokens"), field="usage.completion_tokens"
    )
    if completion_tokens != len(normalized_ids):
        raise ResponseSchemaError(
            "usage.completion_tokens does not equal output token ID count"
        )
    acceptance = meta_info.get("acceptance_counters")
    normalized_spec = normalize_sglang_spec_acceptance(
        meta_info, completion_tokens=completion_tokens
    )
    if acceptance is not None:
        explicit_acceptance = validate_acceptance_counters(acceptance)
        if normalized_spec is not None and explicit_acceptance != normalized_spec:
            raise ResponseSchemaError(
                "acceptance_counters disagree with SGLang spec_* metadata"
            )
    else:
        explicit_acceptance = normalized_spec
    asd = meta_info.get("asd_counters")
    return ParsedResponse(
        model_text=model_text,
        output_token_ids=normalized_ids,
        completion_tokens=completion_tokens,
        acceptance_counters=explicit_acceptance,
        asd_counters=(None if asd is None else validate_asd_counters(asd)),
    )
