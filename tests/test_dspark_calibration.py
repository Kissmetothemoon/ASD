import json
import tempfile
import unittest
from pathlib import Path

from asd.reproduce.dspark.calibration import linear_q25, reduce_calibration
from asd.reproduce.dspark.io import write_json, write_jsonl


def _output(position: int, token_ids: list[int]) -> dict:
    response_id = f"response-{position}"
    return {
        "cohort": "calibration",
        "cohort_position": position,
        "dataset_index": 1000 + position,
        "terminal_status": "succeeded",
        "output_token_ids": token_ids,
        "attempts": [{"status": "success", "raw_response": {"id": response_id}}],
    }


def _trace(position: int, ratio: float) -> dict:
    barrier = position % 5
    value = (5 - barrier) / 5
    return {
        "snapshot_index": 0,
        "proposal_ordinal": position,
        "request_serial": position + 1,
        "rid": f"response-{position % 32}",
        "request_pool_slot": 0,
        "forward_ct": position,
        "barrier_position": barrier,
        "regret": ratio * value,
        "value": value,
        "regret_per_value": ratio,
    }


class CalibrationTests(unittest.TestCase):
    def test_linear_q25_matches_frozen_interpolation(self):
        self.assertAlmostEqual(linear_q25([1.0, 2.0, 3.0, 4.0]), 1.75)
        self.assertEqual(linear_q25([2.0625]), 2.0625)

    def test_linear_q25_rejects_unsorted_or_nonpositive_input(self):
        with self.assertRaises(ValueError):
            linear_q25([])
        with self.assertRaises(ValueError):
            linear_q25([2.0, 1.0])
        with self.assertRaises(ValueError):
            linear_q25([0.0])

    def test_reduce_calibration_proves_b0_and_freezes_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.jsonl"
            b0_path = root / "b0.jsonl"
            trace_path = root / "trace.jsonl"
            counters_path = root / "counters.json"
            output_dir = root / "reduced"
            outputs = [_output(i, [i, i + 1]) for i in range(32)]
            trace = [_trace(i, float(i + 1)) for i in range(32)]
            write_jsonl(native_path, outputs)
            write_jsonl(b0_path, outputs)
            write_jsonl(trace_path, trace)
            write_json(
                counters_path,
                {
                    "status": "PASS",
                    "arm": "native-trace",
                    "trace_rows_seen": 32,
                    "trace_rows_stored": 32,
                    "trace_rows_dropped": 0,
                    "native_acceptance_preserved": True,
                    "cohort_response_ids": [f"response-{i}" for i in range(32)],
                    "trace_scope_proven": True,
                },
            )

            summary = reduce_calibration(
                native_outputs=native_path,
                b0_outputs=b0_path,
                native_trace=trace_path,
                native_counters=counters_path,
                output_dir=output_dir,
            )

            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["positive_value_count"], 32)
            self.assertAlmostEqual(summary["q25"], 8.75)
            self.assertEqual(
                json.loads((output_dir / "asd_config.json").read_text()),
                {
                    "B": 8.75,
                    "g": 8.75,
                    "m": 1,
                    "value_scheme": "normalized_suffix",
                    "block_size": 5,
                },
            )

    def test_reduce_refuses_a_b0_token_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = [_output(i, [i]) for i in range(32)]
            b0 = [_output(i, [i]) for i in range(32)]
            b0[7]["output_token_ids"] = [999]
            write_jsonl(root / "native.jsonl", native)
            write_jsonl(root / "b0.jsonl", b0)
            write_jsonl(root / "trace.jsonl", [_trace(0, 1.0)])
            write_json(root / "counters.json", {})

            with self.assertRaisesRegex(ValueError, "B=0 is not token-identical"):
                reduce_calibration(
                    native_outputs=root / "native.jsonl",
                    b0_outputs=root / "b0.jsonl",
                    native_trace=root / "trace.jsonl",
                    native_counters=root / "counters.json",
                    output_dir=root / "reduced",
                )


if __name__ == "__main__":
    unittest.main()
