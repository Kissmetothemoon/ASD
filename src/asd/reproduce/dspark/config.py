"""Frozen request and dataset configuration for ASD-on-V4 experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any

DATASET_PROVIDER = "Hugging Face"
DATASET_REPO_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SHUFFLE_SEED = 980406
CALIBRATION_COUNT = 32
FORMAL_COUNT = 500
WARMUP_COUNT = 10

DEFAULT_MODEL = "deepseek-v4-flash-dspark"
MODEL_REPO_ID = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
MODEL_REVISION = "62af8fffb2f7030cac4de2f0169f5b8d1101b646"
MODEL_PROVIDER_SNAPSHOT = (
    "bb7ac3172e1a257482d3256d7a720f20ea39ce25625f3cacc1091f59ad43bcae"
)
SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang.git"
SGLANG_VERSION = "0.5.16"
SGLANG_COMMIT = "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"
TRACE_CAPACITY = 65536
TP_SIZE = 8
CONTEXT_LENGTH = 4096
MAX_RUNNING_REQUESTS = 1
MEM_FRACTION_STATIC = 0.80
PROMPT_SUFFIX = (
    r"Please reason step by step, and put your final answer within \boxed{}."
)
TEMPERATURE = 0
TOP_P = 1
MAX_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 600.0
MAX_TOTAL_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0
PROPOSAL_WIDTH = 5

PROTOCOL_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
COUNTER_SCHEMA_VERSION = 1


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's stable JSON representation."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fingerprint_document(value: dict[str, Any]) -> str:
    """Hash a document while excluding its own fingerprint field."""

    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def build_user_content(question: str) -> str:
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a non-empty string")
    return f"{question}\n{PROMPT_SUFFIX}"


def build_chat_request(question: str, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Build the one-message request shared by every experimental arm."""

    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    return {
        "model": model,
        "messages": [{"role": "user", "content": build_user_content(question)}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        # P03 supplies output_token_ids in choices[0].meta_info. Asking for
        # metadata is observational and must not alter decode decisions.
        "return_meta_info": True,
    }


def protocol_config_document() -> dict[str, Any]:
    """Return the immutable, machine-readable P01 protocol."""

    document: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "dataset": {
            "provider": DATASET_PROVIDER,
            "repo_id": DATASET_REPO_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "shuffle": {
                "implementation": (
                    "indices=list(range(test_size)); "
                    "random.Random(980406).shuffle(indices)"
                ),
                "seed": SHUFFLE_SEED,
                "calibration_count": CALIBRATION_COUNT,
                "formal_count": FORMAL_COUNT,
                "warmup_count": WARMUP_COUNT,
                "warmup_selection": "calibration[0:10]",
            },
        },
        "request": {
            "endpoint": "/v1/chat/completions",
            "model": DEFAULT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content_template": f"<original question>\n{PROMPT_SUFFIX}",
                }
            ],
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "stream": False,
            "return_meta_info": True,
        },
        "runner": {
            "concurrency": 1,
            "order": "manifest cohort_position ascending",
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_total_attempts_per_request": MAX_TOTAL_ATTEMPTS,
            "retry_backoff_seconds": RETRY_BACKOFF_SECONDS,
            "retry_time_in_formal_timing": True,
            "terminal_statuses": ["succeeded", "failed"],
        },
        "formal_timing": {
            "clock": "time.monotonic_ns",
            "warmup_count": WARMUP_COUNT,
            "formal_count": FORMAL_COUNT,
            "start_boundary": "immediately before formal request 1 is issued",
            "end_boundary": "immediately after formal request 500 is terminal",
            "excluded": ["service startup", "model load", "warmup requests"],
            "included": ["HTTP", "generation", "queueing", "retry", "retry backoff"],
            "tps_formula": "formal completion_tokens / timed_wall_seconds",
        },
        "response": {
            "raw_response_saved": True,
            "canonical_token_ids_path": ("choices[0].meta_info.output_token_ids"),
            "token_ids_required": True,
            "token_ids_reencoding_forbidden": True,
            "completion_tokens_path": "usage.completion_tokens",
            "completion_tokens_must_equal_token_id_count": True,
            "model_text_path": "choices[0].message.content",
            "answer_rule": "last complete numeric \\\\boxed{...}",
            "reference_rule": "last numeric #### marker",
        },
        "per_request_record_schema": {
            "schema_version": RECORD_SCHEMA_VERSION,
            "required_identity": [
                "cohort",
                "cohort_position",
                "dataset_index",
                "dataset_revision",
                "dataset_fingerprint",
            ],
            "required_request_evidence": [
                "question",
                "reference_response",
                "reference_answer",
                "request_body",
                "attempts",
            ],
            "required_terminal_evidence": [
                "terminal_status",
                "attempt_count",
                "retry_count",
                "request_started_monotonic_ns",
                "terminal_monotonic_ns",
                "request_wall_seconds",
            ],
            "successful_response_evidence": [
                "raw response in successful attempt",
                "model_text",
                "output_token_ids",
                "completion_tokens",
                "extracted_answer",
                "extraction_rule",
                "parse_status",
                "answer_match",
            ],
        },
        "acceptance_counter_schema": {
            "schema_version": COUNTER_SCHEMA_VERSION,
            "location": "normalized from choices[0].meta_info.spec_*",
            "semantics": "per-request delta, never a cumulative server snapshot",
            "fields": {
                "proposal_count": "non-negative integer",
                "draft_tokens_proposed": "non-negative integer",
                "accepted_draft_tokens": "non-negative integer",
                "accepted_draft_tokens_by_position": (
                    f"length-{PROPOSAL_WIDTH} non-negative integer array"
                ),
                "acceptance_length_including_bonus_sum": (
                    "non-negative integer or null"
                ),
                "acceptance_length_including_bonus_count": (
                    "non-negative integer or null"
                ),
            },
        },
        "asd_counter_schema": {
            "schema_version": COUNTER_SCHEMA_VERSION,
            "location": (
                "choices[0].meta_info.asd_counters (optional per-request "
                "delta until P03 instrumentation is enabled)"
            ),
            "semantics": (
                "per-request terminal observation; additive fields are deltas, "
                "active_request_states is a post-terminal gauge; never a "
                "cumulative server snapshot"
            ),
            "fields": {
                "runtime_calls": "non-negative integer",
                "relaxed_tokens": "non-negative integer",
                "budget_spent": "finite non-negative number",
                "budget_hit_requests": "0 or 1",
                "cap_bound_blocks": "non-negative integer",
                "requests_initialized": "non-negative integer",
                "requests_finalized": "non-negative integer",
                "requests_aborted": "non-negative integer",
                "slot_resets": "non-negative integer",
                "active_request_states": (
                    "non-negative post-terminal gauge; each terminal request "
                    "must observe zero"
                ),
                "state_leaks": "non-negative integer; arm-final aggregate must be zero",
            },
        },
    }
    document["fingerprint_sha256"] = fingerprint_document(document)
    return document
