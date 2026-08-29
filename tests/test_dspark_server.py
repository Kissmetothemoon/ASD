import unittest
from collections.abc import Mapping
from typing import Any

from asd.reproduce.dspark.server import (
    SERVER_FIELDS,
    collect_server_evidence,
    extract_asd_snapshots,
)


def _snapshot(mode: str = "calibration") -> dict[str, Any]:
    return {
        "mode": mode,
        "experiment_switches": {
            "ASD_ENABLED": int(mode == "enabled"),
            "SGLANG_DSPARK_ASD_CALIBRATION_TRACE": int(mode == "calibration"),
        },
        "config": None,
        "config_fingerprint": None,
        "gamma": 5,
        "verify_num_draft_tokens": 6,
        "active_request_states": 0,
        "state_leaks": 0,
        "trace_rows_seen": 1,
        "trace_rows_dropped": 0,
        "strict_rejection_trace": [
            {
                "proposal_ordinal": 0,
                "request_serial": 1,
                "rid": "response-0",
                "request_pool_slot": 0,
                "forward_ct": 1,
                "barrier_position": 0,
                "regret": 1.0,
                "value": 1.0,
                "regret_per_value": 1.0,
            }
        ],
    }


def _server_info(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **SERVER_FIELDS,
        "internal_states": [{"dspark_info_record": {"asd": snapshot}}],
    }


class _Client:
    def __init__(self, value: Mapping[str, Any]) -> None:
        self.value = value

    def get(self, url: str, *, timeout: float = 10.0) -> Mapping[str, Any]:
        assert url.endswith("/server_info")
        return self.value


class ServerEvidenceTests(unittest.TestCase):
    def test_extract_snapshots_checks_decode_identity(self):
        info = _server_info(_snapshot())
        snapshots = extract_asd_snapshots(
            info, expected_mode="calibration", expected_config=None
        )
        self.assertEqual(snapshots[0]["gamma"], 5)
        info["tp_size"] = 4
        with self.assertRaisesRegex(ValueError, "decode identity"):
            extract_asd_snapshots(
                info, expected_mode="calibration", expected_config=None
            )

    def test_collect_trace_proves_response_scope(self):
        evidence, trace = collect_server_evidence(
            base_url="http://127.0.0.1:1",
            expected_mode="calibration",
            expected_config=None,
            cohort_response_ids=["response-0"],
            client=_Client(_server_info(_snapshot())),
        )
        self.assertIs(evidence["trace_scope_proven"], True)
        self.assertEqual(evidence["trace_rows_stored"], 1)
        self.assertEqual(trace[0]["snapshot_index"], 0)

    def test_collect_trace_rejects_out_of_scope_response(self):
        with self.assertRaisesRegex(ValueError, "outside the cohort"):
            collect_server_evidence(
                base_url="http://127.0.0.1:1",
                expected_mode="calibration",
                expected_config=None,
                cohort_response_ids=["another-response"],
                client=_Client(_server_info(_snapshot())),
            )


if __name__ == "__main__":
    unittest.main()
