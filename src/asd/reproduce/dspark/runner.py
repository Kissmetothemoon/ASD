"""Sequential, retry-bounded OpenAI-compatible protocol runner."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .answers import extract_model_answer, extract_reference_answer
from .config import (
    DATASET_REVISION,
    DEFAULT_MODEL,
    FORMAL_COUNT,
    MAX_TOTAL_ATTEMPTS,
    RECORD_SCHEMA_VERSION,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    WARMUP_COUNT,
    build_chat_request,
    canonical_json_bytes,
    fingerprint_document,
)
from .io import load_jsonl, sha256_file, write_json
from .schema import ParsedResponse, parse_chat_response
from .summary import recompute_summary

Transport = Callable[[str, dict[str, Any], float], Mapping[str, Any]]
Clock = Callable[[], int]
Sleeper = Callable[[float], None]


class HttpTransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.response_body = response_body


class HttpJsonTransport:
    """Minimal no-proxy JSON transport used by the command-line runner."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read()
                if response.status < 200 or response.status >= 300:
                    raise HttpTransportError(
                        f"unexpected HTTP status {response.status}",
                        http_status=response.status,
                        response_body=raw.decode(errors="replace"),
                    )
        except urllib.error.HTTPError as error:
            raise HttpTransportError(
                f"HTTP request failed with status {error.code}",
                http_status=error.code,
                response_body=error.read().decode(errors="replace"),
            ) from error
        except urllib.error.URLError as error:
            raise HttpTransportError(f"HTTP request failed: {error!r}") from error
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HttpTransportError(
                f"response was not valid JSON: {error!r}",
                http_status=response.status,
                response_body=raw.decode(errors="replace"),
            ) from error
        if not isinstance(value, Mapping):
            raise HttpTransportError("JSON response must be an object")
        return value


def _seconds(start_ns: int, end_ns: int) -> float:
    if end_ns < start_ns:
        raise RuntimeError("monotonic clock moved backwards")
    return (end_ns - start_ns) / 1_000_000_000


class ProtocolRunner:
    """Run requests synchronously and preserve every attempt."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        max_total_attempts: int = MAX_TOTAL_ATTEMPTS,
        retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
        transport: Transport | None = None,
        clock_ns: Clock = time.monotonic_ns,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if not 1 <= max_total_attempts <= MAX_TOTAL_ATTEMPTS:
            raise ValueError(
                f"max_total_attempts must be between 1 and {MAX_TOTAL_ATTEMPTS}"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_total_attempts = max_total_attempts
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.transport = transport or HttpJsonTransport()
        self.clock_ns = clock_ns
        self.sleeper = sleeper

    @staticmethod
    def _sample_value(sample: Mapping[str, Any], key: str) -> Any:
        if key not in sample:
            raise ValueError(f"sample is missing required field {key!r}")
        return sample[key]

    def _validated_reference(
        self, sample: Mapping[str, Any]
    ) -> tuple[str | None, str | None, str]:
        answer = self._sample_value(sample, "answer")
        if not isinstance(answer, str):
            raise TypeError("sample answer must be a string")
        extracted = extract_reference_answer(answer)
        if extracted.value is None:
            raise ValueError("sample answer has no parseable final #### value")
        stored = self._sample_value(sample, "reference_answer")
        if extracted.value != stored:
            raise ValueError(
                "sample reference_answer does not match answer re-extraction"
            )
        return extracted.value, extracted.raw, extracted.rule

    @staticmethod
    def _error_evidence(error: Exception) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error": repr(error),
        }
        if isinstance(error, HttpTransportError):
            evidence["http_status"] = error.http_status
            evidence["response_body"] = error.response_body
        return evidence

    def run_one(
        self,
        sample: Mapping[str, Any],
        *,
        cohort: str,
        cohort_position: int,
    ) -> dict[str, Any]:
        dataset_revision = self._sample_value(sample, "dataset_revision")
        if dataset_revision != DATASET_REVISION:
            raise ValueError(
                f"sample revision {dataset_revision!r} is not pinned revision"
            )
        dataset_fingerprint = self._sample_value(sample, "dataset_fingerprint")
        if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
            raise ValueError("sample dataset_fingerprint must be non-empty")
        question = self._sample_value(sample, "question")
        if not isinstance(question, str) or not question:
            raise ValueError("sample question must be non-empty")
        reference_answer, reference_raw, reference_rule = self._validated_reference(
            sample
        )
        request_body = build_chat_request(question, model=self.model)
        request_started = self.clock_ns()
        attempts: list[dict[str, Any]] = []
        parsed: ParsedResponse | None = None

        for attempt_number in range(1, self.max_total_attempts + 1):
            attempt_started = self.clock_ns()
            raw_response: Mapping[str, Any] | None = None
            try:
                raw_response = self.transport(
                    self.url, request_body, self.timeout_seconds
                )
                parsed = parse_chat_response(raw_response)
                attempt_finished = self.clock_ns()
                attempts.append(
                    {
                        "attempt_number": attempt_number,
                        "status": "success",
                        "started_monotonic_ns": attempt_started,
                        "finished_monotonic_ns": attempt_finished,
                        "wall_seconds": _seconds(attempt_started, attempt_finished),
                        "raw_response": raw_response,
                    }
                )
                break
            except Exception as error:  # noqa: BLE001 - retries preserve arbitrary failures
                attempt_finished = self.clock_ns()
                attempt = {
                    "attempt_number": attempt_number,
                    "status": "error",
                    "started_monotonic_ns": attempt_started,
                    "finished_monotonic_ns": attempt_finished,
                    "wall_seconds": _seconds(attempt_started, attempt_finished),
                    **self._error_evidence(error),
                }
                if raw_response is not None:
                    # A schema failure must retain the exact rejected response.
                    attempt["raw_response"] = raw_response
                if attempt_number < self.max_total_attempts:
                    attempt["retry_delay_seconds_after"] = self.retry_backoff_seconds
                attempts.append(attempt)
                if attempt_number < self.max_total_attempts:
                    self.sleeper(self.retry_backoff_seconds)

        model_extraction = (
            None if parsed is None else extract_model_answer(parsed.model_text)
        )
        terminal_monotonic_ns = self.clock_ns()
        succeeded = parsed is not None
        record: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "cohort": cohort,
            "cohort_position": cohort_position,
            "manifest_cohort": sample.get("cohort"),
            "manifest_cohort_position": sample.get("cohort_position"),
            "dataset_index": self._sample_value(sample, "dataset_index"),
            "dataset_revision": dataset_revision,
            "dataset_fingerprint": dataset_fingerprint,
            "question": question,
            "reference_response": self._sample_value(sample, "answer"),
            "reference_answer": reference_answer,
            "reference_answer_raw": reference_raw,
            "reference_extraction_rule": reference_rule,
            "request_body": request_body,
            "request_started_monotonic_ns": request_started,
            "terminal_monotonic_ns": terminal_monotonic_ns,
            "request_wall_seconds": _seconds(request_started, terminal_monotonic_ns),
            "attempts": attempts,
            "attempt_count": len(attempts),
            "retry_count": max(len(attempts) - 1, 0),
            "request_succeeded": succeeded,
            "terminal_status": "succeeded" if succeeded else "failed",
            "model_text": None if parsed is None else parsed.model_text,
            "output_token_ids": (None if parsed is None else parsed.output_token_ids),
            "completion_tokens": (0 if parsed is None else parsed.completion_tokens),
            "acceptance_counters": (
                None if parsed is None else parsed.acceptance_counters
            ),
            "asd_counters": (None if parsed is None else parsed.asd_counters),
            "extracted_answer": (
                None if model_extraction is None else model_extraction.value
            ),
            "extracted_answer_raw": (
                None if model_extraction is None else model_extraction.raw
            ),
            "extraction_rule": (
                None if model_extraction is None else model_extraction.rule
            ),
        }
        if parsed is None:
            record.update(
                parse_status="not_attempted_after_request_failure",
                answer_match=None,
            )
        elif model_extraction is None or model_extraction.value is None:
            record.update(parse_status="parse_failure", answer_match=None)
        elif reference_answer is None:
            record.update(parse_status="reference_parse_failure", answer_match=None)
        else:
            record.update(
                parse_status="parsed",
                answer_match=model_extraction.value == reference_answer,
            )
        return record

    @staticmethod
    def _new_jsonl(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("x", encoding="utf-8")

    def run_cohort(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        cohort: str,
        output_path: Path,
        expected_count: int | None = None,
    ) -> list[dict[str, Any]]:
        if expected_count is not None and len(samples) != expected_count:
            raise ValueError(
                f"{cohort} requires {expected_count} samples, got {len(samples)}"
            )
        records: list[dict[str, Any]] = []
        with self._new_jsonl(output_path) as output:
            for position, sample in enumerate(samples):
                record = self.run_one(sample, cohort=cohort, cohort_position=position)
                output.write(canonical_json_bytes(record).decode("utf-8"))
                output.flush()
                records.append(record)
        return records

    def run_formal_arm(
        self,
        *,
        warmup_samples: Sequence[Mapping[str, Any]],
        formal_samples: Sequence[Mapping[str, Any]],
        warmup_output: Path,
        formal_output: Path,
        timing_output: Path,
        summary_output: Path,
        acceptance_summary_output: Path | None = None,
        before_formal: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        if len(warmup_samples) != WARMUP_COUNT:
            raise ValueError(
                f"formal arm requires {WARMUP_COUNT} warmups, got {len(warmup_samples)}"
            )
        if len(formal_samples) != FORMAL_COUNT:
            raise ValueError(
                f"formal arm requires {FORMAL_COUNT} formal samples, "
                f"got {len(formal_samples)}"
            )
        self.run_cohort(
            warmup_samples,
            cohort="warmup",
            output_path=warmup_output,
            expected_count=WARMUP_COUNT,
        )
        if before_formal is not None:
            before_formal()

        records: list[dict[str, Any]] = []
        with self._new_jsonl(formal_output) as output:
            formal_started: int | None = None
            formal_finished: int | None = None
            for position, sample in enumerate(formal_samples):
                record = self.run_one(sample, cohort="formal", cohort_position=position)
                if position == 0:
                    formal_started = record["request_started_monotonic_ns"]
                # End immediately when request 500 becomes terminal. The final
                # JSONL serialization is intentionally outside the boundary.
                if position == len(formal_samples) - 1:
                    formal_finished = record["terminal_monotonic_ns"]
                output.write(canonical_json_bytes(record).decode("utf-8"))
                output.flush()
                records.append(record)
        if formal_started is None or formal_finished is None:
            raise AssertionError("formal cohort unexpectedly empty")
        timing = {
            "schema_version": 1,
            "clock": "time.monotonic_ns",
            "warmup_record_count": WARMUP_COUNT,
            "warmup_included": False,
            "formal_record_count": FORMAL_COUNT,
            "formal_start_monotonic_ns": formal_started,
            "formal_end_monotonic_ns": formal_finished,
            "timed_wall_seconds": _seconds(formal_started, formal_finished),
            "start_boundary": ("immediately before formal request 1 is issued"),
            "end_boundary": ("immediately after formal request 500 is terminal"),
            "retry_time_included": True,
            "warmup_output_sha256": sha256_file(warmup_output),
            "formal_output_sha256": sha256_file(formal_output),
        }
        write_json(timing_output, timing)
        summary = recompute_summary(
            records,
            timing=timing,
            expected_count=FORMAL_COUNT,
            expected_cohort="formal",
        )
        summary["source_artifacts"] = {
            "formal_outputs_sha256": sha256_file(formal_output),
            "formal_timing_sha256": sha256_file(timing_output),
        }
        summary["fingerprint_sha256"] = fingerprint_document(summary)
        write_json(summary_output, summary)
        if acceptance_summary_output is not None:
            acceptance_summary = {
                "schema_version": 1,
                "acceptance": summary["acceptance"],
                "asd": summary["asd"],
                "source_artifacts": summary["source_artifacts"],
            }
            acceptance_summary["fingerprint_sha256"] = fingerprint_document(
                acceptance_summary
            )
            write_json(acceptance_summary_output, acceptance_summary)
        return summary


def run_formal_from_files(
    runner: ProtocolRunner,
    *,
    calibration_path: Path,
    formal_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    calibration = load_jsonl(calibration_path)
    formal = load_jsonl(formal_path)
    return runner.run_formal_arm(
        warmup_samples=calibration[:WARMUP_COUNT],
        formal_samples=formal,
        warmup_output=artifact_dir / "warmup_outputs.jsonl",
        formal_output=artifact_dir / "formal_outputs.jsonl",
        timing_output=artifact_dir / "formal_timing.json",
        summary_output=artifact_dir / "answer_summary.json",
        acceptance_summary_output=artifact_dir / "acceptance_summary.json",
    )
