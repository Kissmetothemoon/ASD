from __future__ import annotations

import json
import random
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from asd.reproduce.dspark.answers import (
    extract_model_answer,
    extract_reference_answer,
    normalize_number,
)
from asd.reproduce.dspark.config import (
    DATASET_REVISION,
    PROMPT_SUFFIX,
    build_chat_request,
    fingerprint_document,
    protocol_config_document,
)
from asd.reproduce.dspark.dataset import (
    CALIBRATION_JSONL_NAME,
    DATASET_MANIFEST_NAME,
    FORMAL_JSONL_NAME,
    PROTOCOL_CONFIG_NAME,
    materialize_dataset,
    selected_indices,
    verify_materialized_dataset,
)
from asd.reproduce.dspark.io import load_jsonl, sha256_file
from asd.reproduce.dspark.runner import ProtocolRunner
from asd.reproduce.dspark.schema import (
    ResponseSchemaError,
    parse_chat_response,
)
from asd.reproduce.dspark.summary import recompute_summary


def make_sample(
    dataset_index: int,
    *,
    cohort: str,
    cohort_position: int,
    fingerprint: str = "fixture-fingerprint",
) -> dict[str, Any]:
    answer = str(dataset_index)
    return {
        "schema_version": 1,
        "cohort": cohort,
        "cohort_position": cohort_position,
        "dataset_index": dataset_index,
        "dataset_revision": DATASET_REVISION,
        "dataset_fingerprint": fingerprint,
        "question": f"Question {dataset_index}?",
        "answer": f"fixture reasoning\n#### {answer}",
        "reference_answer": answer,
        "reference_answer_raw": answer,
        "reference_extraction_rule": "last_hashes",
        "user_content": f"Question {dataset_index}?\n{PROMPT_SUFFIX}",
    }


def acceptance_counters() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "proposal_count": 1,
        "draft_tokens_proposed": 5,
        "accepted_draft_tokens": 1,
        "accepted_draft_tokens_by_position": [1, 0, 0, 0, 0],
        "acceptance_length_including_bonus_sum": 2,
        "acceptance_length_including_bonus_count": 1,
    }


def asd_counters() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runtime_calls": 1,
        "relaxed_tokens": 1,
        "budget_spent": 0.25,
        "budget_hit_requests": 1,
        "cap_bound_blocks": 0,
        "requests_initialized": 1,
        "requests_finalized": 1,
        "requests_aborted": 0,
        "slot_resets": 1,
        "active_request_states": 0,
        "state_leaks": 0,
    }


def response_for_answer(
    answer: str,
    *,
    token_ids: list[int] | None = None,
    include_counters: bool = True,
) -> dict[str, Any]:
    token_ids = [101] if token_ids is None else token_ids
    meta_info: dict[str, Any] = {"output_token_ids": token_ids}
    if include_counters:
        meta_info["acceptance_counters"] = acceptance_counters()
        meta_info["asd_counters"] = asd_counters()
    return {
        "id": "fixture",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": rf"work complete \boxed{{{answer}}}",
                },
                "finish_reason": "stop",
                "meta_info": meta_info,
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": len(token_ids),
            "total_tokens": 7 + len(token_ids),
        },
    }


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1_000_000_000)

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


class DeterministicTransport:
    def __init__(
        self,
        clock: FakeClock,
        *,
        fail_call_numbers: set[int] | None = None,
    ) -> None:
        self.clock = clock
        self.fail_call_numbers = fail_call_numbers or set()
        self.call_count = 0
        self.observed_questions: list[int] = []
        self.active = 0
        self.max_active = 0

    def __call__(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.call_count += 1
            self.clock.advance(1)
            content = payload["messages"][0]["content"]
            match = re.search(r"Question (\d+)\?", content)
            if match is None:
                raise AssertionError("fixture request prompt did not match")
            answer = int(match.group(1))
            self.observed_questions.append(answer)
            if self.call_count in self.fail_call_numbers:
                raise OSError(f"fixture failure call {self.call_count}")
            return response_for_answer(str(answer))
        finally:
            self.active -= 1


class FrozenConfigTests(unittest.TestCase):
    def test_prompt_and_request_are_exact(self) -> None:
        question = "Janet has 3 apples."
        request = build_chat_request(question)
        self.assertEqual(
            request["messages"],
            [
                {
                    "role": "user",
                    "content": f"{question}\n{PROMPT_SUFFIX}",
                }
            ],
        )
        self.assertNotIn("system", request)
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["top_p"], 1)
        self.assertEqual(request["max_tokens"], 512)
        self.assertFalse(request["stream"])
        self.assertTrue(request["return_meta_info"])

    def test_protocol_fingerprint_recomputes(self) -> None:
        document = protocol_config_document()
        self.assertEqual(document["fingerprint_sha256"], fingerprint_document(document))
        self.assertEqual(
            document["response"]["canonical_token_ids_path"],
            "choices[0].meta_info.output_token_ids",
        )
        self.assertTrue(document["response"]["token_ids_reencoding_forbidden"])


class AnswerParsingTests(unittest.TestCase):
    def test_reference_uses_last_hashes_marker(self) -> None:
        result = extract_reference_answer("old #### 2\nfinal #### $1,200.00")
        self.assertEqual(result.value, "1200")
        self.assertEqual(result.rule, "last_hashes")

    def test_model_uses_last_complete_boxed_answer_only(self) -> None:
        result = extract_model_answer(r"first \boxed{8}; final \boxed{-9.00}")
        self.assertEqual(result.value, "-9")
        self.assertEqual(result.raw, "-9.00")
        self.assertEqual(result.rule, "last_boxed")
        self.assertIsNone(extract_model_answer("Final answer: 9").value)
        self.assertIsNone(extract_model_answer(r"\boxed{\frac{1}{2}}").value)

    def test_numeric_normalization_is_strict(self) -> None:
        self.assertEqual(normalize_number(r"\$1,020.500"), "1020.5")
        self.assertEqual(normalize_number("-0.000"), "0")
        self.assertIsNone(normalize_number("approximately 4"))
        self.assertIsNone(normalize_number("1/2"))


class DatasetMaterializationTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self) -> None:
        calibration, formal = selected_indices(1319)
        expected = list(range(1319))
        random.Random(980406).shuffle(expected)
        self.assertEqual(calibration, expected[:32])
        self.assertEqual(formal, expected[32:532])
        self.assertEqual(len(calibration), 32)
        self.assertEqual(len(formal), 500)
        self.assertFalse(set(calibration) & set(formal))

    def test_manifest_and_jsonl_are_reproducible(self) -> None:
        rows = [
            {
                "question": f"Question {index}?",
                "answer": f"reasoning\n#### {index}",
            }
            for index in range(600)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = root / "generator.py"
            generator.write_text("pinned generator fixture\n", encoding="utf-8")
            first = root / "first"
            second = root / "second"
            for output in (first, second):
                materialize_dataset(
                    rows,
                    resolved_revision=DATASET_REVISION,
                    dataset_fingerprint="fixture-dataset-fingerprint",
                    output_dir=output,
                    generator_path=generator,
                    repository_root=root,
                )
            for name in (
                DATASET_MANIFEST_NAME,
                CALIBRATION_JSONL_NAME,
                FORMAL_JSONL_NAME,
                PROTOCOL_CONFIG_NAME,
            ):
                self.assertEqual(
                    (first / name).read_bytes(), (second / name).read_bytes()
                )
            manifest = json.loads(
                (first / DATASET_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            calibration = load_jsonl(first / CALIBRATION_JSONL_NAME)
            formal = load_jsonl(first / FORMAL_JSONL_NAME)
            self.assertEqual(len(calibration), 32)
            self.assertEqual(len(formal), 500)
            self.assertTrue(manifest["selection"]["cohorts_disjoint"])
            self.assertFalse(
                {row["dataset_index"] for row in calibration}
                & {row["dataset_index"] for row in formal}
            )
            self.assertEqual(
                sha256_file(first / CALIBRATION_JSONL_NAME),
                manifest["artifacts"][CALIBRATION_JSONL_NAME]["sha256"],
            )
            self.assertEqual(
                sha256_file(first / FORMAL_JSONL_NAME),
                manifest["artifacts"][FORMAL_JSONL_NAME]["sha256"],
            )
            self.assertTrue(
                all(
                    row["user_content"] == f"{row['question']}\n{PROMPT_SUFFIX}"
                    for row in calibration + formal
                )
            )
            verification = verify_materialized_dataset(first, repository_root=root)
            self.assertEqual(verification["status"], "PASS")


class ResponseSchemaTests(unittest.TestCase):
    def test_token_ids_are_required_and_never_reencoded(self) -> None:
        response = response_for_answer("4")
        del response["choices"][0]["meta_info"]["output_token_ids"]
        with self.assertRaisesRegex(ResponseSchemaError, "output_token_ids"):
            parse_chat_response(response)

    def test_completion_tokens_must_match_token_ids(self) -> None:
        response = response_for_answer("4", token_ids=[1, 2])
        response["usage"]["completion_tokens"] = 1
        with self.assertRaisesRegex(ResponseSchemaError, "does not equal"):
            parse_chat_response(response)

    def test_counter_shape_is_validated(self) -> None:
        response = response_for_answer("4")
        response["choices"][0]["meta_info"]["acceptance_counters"][
            "accepted_draft_tokens_by_position"
        ] = [1, 0]
        with self.assertRaisesRegex(ResponseSchemaError, "proposal width"):
            parse_chat_response(response)

    def test_sglang_spec_fields_are_normalized_and_cross_checked(self) -> None:
        response = response_for_answer(
            "4", token_ids=[101, 102], include_counters=False
        )
        response["choices"][0]["meta_info"].update(
            {
                "spec_verify_ct": 1,
                "spec_num_proposed_drafts": 5,
                "spec_proposed_drafts": 5,
                "spec_num_correct_drafts": 1,
                "spec_accepted_drafts": 1,
                "spec_correct_drafts_histogram": [0, 1, 0, 0, 0, 0],
                "spec_accept_histogram": [0, 1, 0, 0, 0, 0],
                "spec_accept_rate": 0.2,
                "spec_accept_length": 2.0,
                "completion_tokens": 2,
            }
        )
        parsed = parse_chat_response(response)
        self.assertEqual(parsed.acceptance_counters, acceptance_counters())

    def test_partial_sglang_spec_fields_are_rejected(self) -> None:
        response = response_for_answer("4", include_counters=False)
        response["choices"][0]["meta_info"]["spec_verify_ct"] = 1
        with self.assertRaisesRegex(ResponseSchemaError, "incomplete"):
            parse_chat_response(response)


class RunnerRetryTests(unittest.TestCase):
    def test_schema_failures_retry_three_total_and_preserve_raw(self) -> None:
        clock = FakeClock()
        attempts = 0

        def transport(
            url: str, payload: dict[str, Any], timeout: float
        ) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            clock.advance(1)
            response = response_for_answer("7")
            if attempts < 3:
                del response["choices"][0]["meta_info"]["output_token_ids"]
            return response

        runner = ProtocolRunner(
            base_url="http://fixture",
            transport=transport,
            clock_ns=clock,
            sleeper=clock.sleep,
        )
        record = runner.run_one(
            make_sample(7, cohort="calibration", cohort_position=0),
            cohort="calibration",
            cohort_position=0,
        )
        self.assertEqual(record["terminal_status"], "succeeded")
        self.assertEqual(record["attempt_count"], 3)
        self.assertEqual(record["retry_count"], 2)
        self.assertTrue(
            all("raw_response" in attempt for attempt in record["attempts"])
        )
        self.assertEqual(record["request_wall_seconds"], 5.0)

    def test_permanent_error_becomes_terminal_after_three_attempts(self) -> None:
        clock = FakeClock()

        def transport(
            url: str, payload: dict[str, Any], timeout: float
        ) -> dict[str, Any]:
            clock.advance(1)
            raise OSError("fixture unavailable")

        runner = ProtocolRunner(
            base_url="http://fixture",
            transport=transport,
            clock_ns=clock,
            sleeper=clock.sleep,
        )
        record = runner.run_one(
            make_sample(8, cohort="calibration", cohort_position=0),
            cohort="calibration",
            cohort_position=0,
        )
        self.assertEqual(record["terminal_status"], "failed")
        self.assertEqual(record["attempt_count"], 3)
        self.assertEqual(record["completion_tokens"], 0)
        self.assertIsNone(record["answer_match"])


class FormalTimingAndSummaryTests(unittest.TestCase):
    def test_10_warmups_are_fully_excluded_and_500_terminal_recompute(self) -> None:
        clock = FakeClock()
        # Call 11 is formal request 1's first attempt. It fails once: one
        # extra request second plus one retry-backoff second must be timed.
        transport = DeterministicTransport(clock, fail_call_numbers={11})
        runner = ProtocolRunner(
            base_url="http://fixture",
            transport=transport,
            clock_ns=clock,
            sleeper=clock.sleep,
        )
        warmups = [
            make_sample(
                index,
                cohort="calibration",
                cohort_position=index,
            )
            for index in range(10)
        ]
        formal = [
            make_sample(
                1000 + index,
                cohort="formal",
                cohort_position=index,
            )
            for index in range(500)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            warmup_path = root / "warmup_outputs.jsonl"
            formal_path = root / "formal_outputs.jsonl"
            timing_path = root / "formal_timing.json"
            summary_path = root / "answer_summary.json"
            summary = runner.run_formal_arm(
                warmup_samples=warmups,
                formal_samples=formal,
                warmup_output=warmup_path,
                formal_output=formal_path,
                timing_output=timing_path,
                summary_output=summary_path,
            )
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            self.assertEqual(timing["formal_start_monotonic_ns"], 10_000_000_000)
            self.assertEqual(timing["formal_end_monotonic_ns"], 512_000_000_000)
            self.assertEqual(timing["timed_wall_seconds"], 502.0)
            self.assertFalse(timing["warmup_included"])
            self.assertEqual(len(load_jsonl(warmup_path)), 10)
            formal_records = load_jsonl(formal_path)
            self.assertEqual(len(formal_records), 500)
            self.assertTrue(summary["all_terminal"])
            self.assertEqual(summary["terminal_requests"], 500)
            self.assertEqual(summary["success_requests"], 500)
            self.assertEqual(summary["failed_requests"], 0)
            self.assertEqual(summary["retry_attempts"], 1)
            self.assertEqual(summary["completion_tokens"], 500)
            self.assertEqual(summary["timing"]["end_to_end_output_tps"], 500 / 502)
            self.assertEqual(summary["acceptance"]["proposal_count"], 500)
            self.assertEqual(
                summary["acceptance"]["mean_accepted_draft_tokens_per_proposal"],
                1.0,
            )
            self.assertEqual(summary["asd"]["runtime_calls"], 500)
            self.assertEqual(summary["asd"]["budget_spent"], 125.0)
            self.assertEqual(transport.max_active, 1)
            self.assertEqual(transport.observed_questions[:10], list(range(10)))
            self.assertEqual(transport.observed_questions[10:12], [1000, 1000])
            recomputed = recompute_summary(
                formal_records,
                timing=timing,
                expected_count=500,
                expected_cohort="formal",
            )
            recomputed["source_artifacts"] = {
                "formal_outputs_sha256": sha256_file(formal_path),
                "formal_timing_sha256": sha256_file(timing_path),
            }
            recomputed["fingerprint_sha256"] = fingerprint_document(recomputed)
            self.assertEqual(
                recomputed,
                json.loads(summary_path.read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
