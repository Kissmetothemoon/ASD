"""SGLang DSpark server lifecycle and experiment-counter controls."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .config import (
    CONTEXT_LENGTH,
    MAX_RUNNING_REQUESTS,
    MEM_FRACTION_STATIC,
    TP_SIZE,
    TRACE_CAPACITY,
)
from .io import write_json

SERVER_FIELDS = {
    "tp_size": TP_SIZE,
    "speculative_algorithm": "DSPARK",
    "speculative_dspark_block_size": 5,
    "moe_runner_backend": "flashinfer_mxfp4",
    "speculative_moe_runner_backend": "flashinfer_mxfp4",
    "context_length": CONTEXT_LENGTH,
    "max_running_requests": MAX_RUNNING_REQUESTS,
    "mem_fraction_static": MEM_FRACTION_STATIC,
    "disable_cuda_graph": True,
    "disable_overlap_schedule": True,
    "disable_radix_cache": True,
}


class JsonClient:
    """Small no-proxy client for SGLang control endpoints."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, url: str, *, timeout: float = 10.0) -> Any:
        return self._request(url, method="GET", timeout=timeout)

    def status(self, url: str, *, timeout: float = 10.0) -> int:
        request = urllib.request.Request(url, method="GET")
        try:
            with self._opener.open(request, timeout=timeout) as response:
                response.read()
                return int(response.status)
        except urllib.error.HTTPError as error:
            return int(error.code)
        except urllib.error.URLError as error:
            raise RuntimeError(f"GET {url} failed: {error!r}") from error

    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout: float = 10.0
    ) -> Any:
        return self._request(url, method="POST", payload=payload, timeout=timeout)

    def _request(
        self,
        url: str,
        *,
        method: str,
        timeout: float,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method=method,
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"{method} {url} failed with HTTP {error.code}: "
                + error.read().decode(errors="replace")
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"{method} {url} failed: {error!r}") from error
        try:
            return json.loads(raw) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{method} {url} did not return JSON") from error


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def extract_asd_snapshots(
    server_info: Mapping[str, Any],
    *,
    expected_mode: str,
    expected_config: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Validate the live decode identity and return all ASD snapshots."""

    mismatches = {
        field: {"expected": expected, "observed": server_info.get(field)}
        for field, expected in SERVER_FIELDS.items()
        if server_info.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"/server_info decode identity mismatch: {mismatches}")
    states = server_info.get("internal_states")
    if not isinstance(states, list) or not states:
        raise ValueError("/server_info has no internal_states")
    snapshots: list[dict[str, Any]] = []
    expected_switches = {
        "ASD_ENABLED": int(expected_mode == "enabled"),
        "SGLANG_DSPARK_ASD_CALIBRATION_TRACE": int(expected_mode == "calibration"),
    }
    for position, state in enumerate(states):
        info = state.get("dspark_info_record") if isinstance(state, Mapping) else None
        snapshot = info.get("asd") if isinstance(info, Mapping) else None
        if not isinstance(snapshot, Mapping):
            raise TypeError(f"internal state {position} has no ASD snapshot")
        normalized = dict(snapshot)
        if (
            normalized.get("mode") != expected_mode
            or normalized.get("experiment_switches") != expected_switches
            or normalized.get("config") != expected_config
            or normalized.get("gamma") != 5
            or normalized.get("verify_num_draft_tokens") != 6
        ):
            raise ValueError(f"ASD snapshot {position} identity mismatch")
        _nonnegative_int(
            normalized.get("active_request_states"), field="active_request_states"
        )
        _nonnegative_int(normalized.get("state_leaks"), field="state_leaks")
        snapshots.append(normalized)
    return snapshots


def collect_server_evidence(
    *,
    base_url: str,
    expected_mode: str,
    expected_config: Mapping[str, Any] | None,
    cohort_response_ids: list[str] | None = None,
    require_calibration_trace: bool = True,
    client: JsonClient | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect one post-cohort counter snapshot and flatten calibration rows."""

    getter = client or JsonClient()
    server_info = getter.get(base_url.rstrip("/") + "/server_info", timeout=300)
    if not isinstance(server_info, Mapping):
        raise TypeError("/server_info response must be an object")
    snapshots = extract_asd_snapshots(
        server_info,
        expected_mode=expected_mode,
        expected_config=expected_config,
    )
    trace: list[dict[str, Any]] = []
    trace_seen = 0
    trace_dropped = 0
    for snapshot_index, snapshot in enumerate(snapshots):
        if snapshot["active_request_states"] != 0 or snapshot["state_leaks"] != 0:
            raise ValueError(f"ASD snapshot {snapshot_index} leaked request state")
        trace_seen += _nonnegative_int(
            snapshot.get("trace_rows_seen"), field="trace_rows_seen"
        )
        trace_dropped += _nonnegative_int(
            snapshot.get("trace_rows_dropped"), field="trace_rows_dropped"
        )
        rows = snapshot.get("strict_rejection_trace")
        if not isinstance(rows, list) or any(
            not isinstance(row, Mapping) for row in rows
        ):
            raise ValueError("strict_rejection_trace must be an array of objects")
        trace.extend({"snapshot_index": snapshot_index, **row} for row in rows)
    trace_ids = {row.get("rid") for row in trace}
    scoped = cohort_response_ids is not None and trace_ids.issubset(
        set(cohort_response_ids)
    )
    if cohort_response_ids is not None and not scoped:
        raise ValueError("calibration trace contains response ids outside the cohort")
    evidence = {
        "schema_version": 1,
        "status": "PASS",
        "arm": "native-trace" if expected_mode == "calibration" else expected_mode,
        "source_endpoint": "/server_info",
        "server_config": {field: server_info[field] for field in SERVER_FIELDS},
        "internal_states": server_info["internal_states"],
        "asd_snapshots": snapshots,
        "trace_capacity": TRACE_CAPACITY if expected_mode == "calibration" else None,
        "trace_rows_seen": trace_seen,
        "trace_rows_stored": len(trace),
        "trace_rows_dropped": trace_dropped,
        "native_acceptance_preserved": expected_mode == "calibration",
        "cohort_response_ids": cohort_response_ids,
        "trace_scope_proven": scoped,
    }
    if (
        expected_mode == "calibration"
        and require_calibration_trace
        and (trace_seen <= 0 or trace_dropped)
    ):
        raise ValueError("calibration trace is empty or dropped rows")
    return evidence, trace


def clear_server_metrics(
    *,
    base_url: str,
    expected_mode: str,
    expected_config: Mapping[str, Any] | None,
    timeout_seconds: float = 120.0,
    client: JsonClient | None = None,
) -> dict[str, Any]:
    """Wait for quiescence, clear DSpark counters, and prove exact zero."""

    http = client or JsonClient()
    deadline = time.monotonic() + timeout_seconds
    while True:
        info = http.get(base_url.rstrip("/") + "/server_info")
        snapshots = extract_asd_snapshots(
            info,
            expected_mode=expected_mode,
            expected_config=expected_config,
        )
        if all(snapshot["active_request_states"] == 0 for snapshot in snapshots):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("server did not become quiescent before counter clear")
        time.sleep(1)
    response = http.post(
        base_url.rstrip("/") + "/set_internal_state",
        {"server_args": {"dspark_clear_info_records": 1}},
    )
    if response != [True]:
        raise RuntimeError(f"SGLang rejected counter clear: {response!r}")
    evidence, trace = collect_server_evidence(
        base_url=base_url,
        expected_mode=expected_mode,
        expected_config=expected_config,
        require_calibration_trace=False,
        client=http,
    )
    nonzero: list[dict[str, Any]] = []
    for index, snapshot in enumerate(evidence["asd_snapshots"]):
        dirty = {
            key: value
            for key, value in snapshot.items()
            if key
            in {
                "proposals",
                "draft_tokens_verifiable",
                "strict_accepted_draft_tokens",
                "asd_accepted_draft_tokens",
                "relaxed_mismatches",
                "regret_charged",
                "cap_trim_lens",
                "budget_exhaustion_events",
                "requests_initialized",
                "requests_finished",
                "requests_non_natural",
                "slot_reuse_resets",
                "active_request_states",
                "state_leaks",
                "trace_rows_seen",
                "trace_rows_dropped",
            }
            and value != 0
        }
        if dirty:
            nonzero.append({"snapshot_index": index, "fields": dirty})
    if nonzero or trace:
        raise RuntimeError(f"counter clear did not produce exact zero: {nonzero}")
    return {"schema_version": 1, "status": "PASS", "response": response}


def _mode_environment(mode: str, config_path: Path | None) -> dict[str, str]:
    if mode not in {"disabled", "calibration", "enabled"}:
        raise ValueError(f"unknown ASD server mode: {mode}")
    environment = {
        "ASD_ENABLED": "1" if mode == "enabled" else "0",
        "SGLANG_DSPARK_ASD_CALIBRATION_TRACE": ("1" if mode == "calibration" else "0"),
    }
    if mode == "calibration":
        environment["SGLANG_DSPARK_ASD_TRACE_CAPACITY"] = str(TRACE_CAPACITY)
    if mode == "enabled":
        if config_path is None:
            raise ValueError("enabled mode requires an ASD config path")
        environment["SGLANG_DSPARK_ASD_CONFIG_PATH"] = str(config_path.resolve())
    elif config_path is not None:
        raise ValueError("only enabled mode accepts an ASD config path")
    return environment


def server_command(
    *, python: Path, model_path: Path, host: str, port: int
) -> list[str]:
    return [
        str(python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--served-model-name",
        "deepseek-v4-flash-dspark",
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(TP_SIZE),
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-dspark-block-size",
        "5",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--speculative-moe-runner-backend",
        "flashinfer_mxfp4",
        "--context-length",
        str(CONTEXT_LENGTH),
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
        "--mem-fraction-static",
        str(MEM_FRACTION_STATIC),
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
    ]


class SGLangServer(AbstractContextManager["SGLangServer"]):
    """Own exactly one launched server process group."""

    def __init__(
        self,
        *,
        python: Path,
        model_path: Path,
        artifact_dir: Path,
        mode: str,
        config_path: Path | None,
        host: str = "127.0.0.1",
        port: int = 31066,
        startup_timeout_seconds: float = 3600,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.command = server_command(
            python=python, model_path=model_path, host=host, port=port
        )
        self.mode_env = _mode_environment(mode, config_path)
        self.artifact_dir = artifact_dir
        self.startup_timeout_seconds = startup_timeout_seconds
        self.process: subprocess.Popen[bytes] | None = None
        self._log = None

    def __enter__(self) -> SGLangServer:  # noqa: PYI034 - Python 3.10 compatible
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            self.artifact_dir / "server_launch.json",
            {"command": self.command, "experiment_environment": self.mode_env},
            immutable=True,
        )
        self._log = (self.artifact_dir / "server.log").open("xb")
        environment = os.environ.copy()
        environment.update(self.mode_env)
        self.process = subprocess.Popen(
            self.command,
            env=environment,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        client = JsonClient()
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "SGLang exited during startup with code "
                        f"{self.process.returncode}; see "
                        f"{self.artifact_dir / 'server.log'}"
                    )
                try:
                    if client.status(self.base_url + "/health", timeout=5) == 200:
                        return self
                except RuntimeError:
                    pass
                time.sleep(5)
            raise TimeoutError("SGLang did not become ready before startup timeout")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        process = self.process
        try:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        finally:
            if self._log is not None:
                self._log.close()
