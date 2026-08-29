"""End-to-end orchestration for the public DSpark reproduction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .asd_config import DSparkASDConfig
from .calibration import reduce_calibration
from .config import (
    CALIBRATION_COUNT,
    DATASET_CONFIG,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    DEFAULT_MODEL,
    MODEL_REPO_ID,
    MODEL_REVISION,
)
from .dataset import (
    CALIBRATION_JSONL_NAME,
    FORMAL_JSONL_NAME,
    materialize_dataset,
    verify_materialized_dataset,
)
from .io import load_jsonl, sha256_file, write_json, write_jsonl
from .runner import ProtocolRunner
from .server import (
    SGLangServer,
    clear_server_metrics,
    collect_server_evidence,
)
from .summary import recompute_summary

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
COMMITTED_B0_CONFIG = REPOSITORY_ROOT / "configs" / "deepseek_v4_flash_dspark_b0.json"
COMMITTED_ASD_CONFIG = REPOSITORY_ROOT / "configs" / "deepseek_v4_flash_dspark_asd.json"


def _require_source_assets() -> None:
    if not COMMITTED_B0_CONFIG.is_file() or not COMMITTED_ASD_CONFIG.is_file():
        raise RuntimeError(
            "full DSpark reproduction must be run from an ASD source checkout"
        )


def download_model(output_dir: Path) -> Path:
    """Download the exact public model revision into a local directory."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "download-model requires the reproduction optional dependencies"
        ) from error
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        local_dir=output_dir,
    )
    identity = {
        "schema_version": 1,
        "repo_id": MODEL_REPO_ID,
        "requested_revision": MODEL_REVISION,
        "local_path": str(Path(resolved).resolve()),
    }
    write_json(output_dir / "asd_model_identity.json", identity)
    return Path(resolved)


def prepare_data(output_dir: Path) -> dict[str, Any]:
    """Fetch the pinned GSM8K revision and materialize the 32/500 split."""

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "prepare-data requires the reproduction optional dependencies"
        ) from error
    resolved_revision = (
        HfApi().dataset_info(DATASET_REPO_ID, revision=DATASET_REVISION).sha
    )
    if resolved_revision != DATASET_REVISION:
        raise RuntimeError(
            f"Hugging Face resolved {resolved_revision!r}, "
            f"expected {DATASET_REVISION!r}"
        )
    dataset = load_dataset(
        DATASET_REPO_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    rows = [dict(row) for row in dataset]
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError("datasets did not expose a non-empty fingerprint")
    manifest = materialize_dataset(
        rows,
        resolved_revision=resolved_revision,
        dataset_fingerprint=fingerprint,
        output_dir=output_dir,
        generator_path=Path(__file__).resolve(),
        repository_root=REPOSITORY_ROOT,
    )
    verify_materialized_dataset(output_dir, repository_root=REPOSITORY_ROOT)
    return manifest


def _response_ids(records: list[Mapping[str, Any]]) -> list[str]:
    response_ids: list[str] = []
    for position, record in enumerate(records):
        attempts = record.get("attempts")
        successes = (
            [
                attempt
                for attempt in attempts
                if isinstance(attempt, Mapping) and attempt.get("status") == "success"
            ]
            if isinstance(attempts, list)
            else []
        )
        response = successes[-1].get("raw_response") if len(successes) == 1 else None
        response_id = response.get("id") if isinstance(response, Mapping) else None
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"record {position} has no unique successful response id")
        response_ids.append(response_id)
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("response ids are not unique")
    return response_ids


def run_calibration_arm(
    *,
    arm: str,
    python: Path,
    model_path: Path,
    data_dir: Path,
    artifact_dir: Path,
    port: int,
) -> dict[str, Any]:
    """Run either native trace capture or the strict B=0 equivalence arm."""

    _require_source_assets()
    if arm not in {"native-trace", "b0"}:
        raise ValueError("calibration arm must be native-trace or b0")
    samples = load_jsonl(data_dir / CALIBRATION_JSONL_NAME)
    mode = "calibration" if arm == "native-trace" else "enabled"
    config_path = None if arm == "native-trace" else COMMITTED_B0_CONFIG
    expected_config = (
        None
        if config_path is None
        else DSparkASDConfig.from_json(config_path).to_mapping()
    )
    with SGLangServer(
        python=python,
        model_path=model_path,
        artifact_dir=artifact_dir,
        mode=mode,
        config_path=config_path,
        port=port,
    ) as server:
        clear_evidence = clear_server_metrics(
            base_url=server.base_url,
            expected_mode=mode,
            expected_config=expected_config,
        )
        write_json(artifact_dir / "pre_cohort_clear.json", clear_evidence)
        runner = ProtocolRunner(base_url=server.base_url, model=DEFAULT_MODEL)
        records = runner.run_cohort(
            samples,
            cohort="calibration",
            output_path=artifact_dir / "outputs.jsonl",
            expected_count=CALIBRATION_COUNT,
        )
        summary = recompute_summary(
            records,
            expected_count=CALIBRATION_COUNT,
            expected_cohort="calibration",
        )
        if summary["success_requests"] != CALIBRATION_COUNT:
            raise RuntimeError("calibration requires all 32 requests to succeed")
        counters, trace = collect_server_evidence(
            base_url=server.base_url,
            expected_mode=mode,
            expected_config=expected_config,
            cohort_response_ids=_response_ids(records),
            require_calibration_trace=arm == "native-trace",
        )
        if arm == "b0":
            for snapshot in counters["asd_snapshots"]:
                if (
                    snapshot["relaxed_mismatches"] != 0
                    or float(snapshot["regret_charged"]) != 0.0
                    or snapshot["asd_accepted_draft_tokens"]
                    != snapshot["strict_accepted_draft_tokens"]
                ):
                    raise RuntimeError("B=0 did not preserve strict acceptance")
        write_json(artifact_dir / "counters.json", counters, immutable=True)
        write_jsonl(artifact_dir / "trace.jsonl", trace, immutable=True)
        result = {
            "schema_version": 1,
            "status": "PASS",
            "arm": arm,
            "sample_count": CALIBRATION_COUNT,
            "completion_tokens": summary["completion_tokens"],
            "outputs_sha256": sha256_file(artifact_dir / "outputs.jsonl"),
            "trace_rows_seen": counters["trace_rows_seen"],
            "trace_rows_stored": counters["trace_rows_stored"],
            "trace_rows_dropped": counters["trace_rows_dropped"],
            "trace_scope_proven": counters["trace_scope_proven"],
        }
        write_json(artifact_dir / "summary.json", result, immutable=True)
        return result


def reduce_calibration_run(calibration_dir: Path) -> dict[str, Any]:
    _require_source_assets()
    result = reduce_calibration(
        native_outputs=calibration_dir / "native-trace" / "outputs.jsonl",
        b0_outputs=calibration_dir / "b0" / "outputs.jsonl",
        native_trace=calibration_dir / "native-trace" / "trace.jsonl",
        native_counters=calibration_dir / "native-trace" / "counters.json",
        output_dir=calibration_dir / "reduced",
    )
    generated = DSparkASDConfig.from_json(
        calibration_dir / "reduced" / "asd_config.json"
    )
    committed = DSparkASDConfig.from_json(COMMITTED_ASD_CONFIG)
    if generated != committed:
        raise RuntimeError(
            "fresh calibration differs from "
            "configs/deepseek_v4_flash_dspark_asd.json: "
            f"generated={generated.to_mapping()} committed={committed.to_mapping()}"
        )
    return result


def run_formal_arm(
    *,
    arm: str,
    python: Path,
    model_path: Path,
    data_dir: Path,
    artifact_dir: Path,
    asd_config_path: Path,
    port: int,
) -> dict[str, Any]:
    """Run one warmup-10 plus timed formal-500 arm."""

    _require_source_assets()
    if arm not in {"native", "asd"}:
        raise ValueError("formal arm must be native or asd")
    calibration = load_jsonl(data_dir / CALIBRATION_JSONL_NAME)
    formal = load_jsonl(data_dir / FORMAL_JSONL_NAME)
    mode = "disabled" if arm == "native" else "enabled"
    config_path = None if arm == "native" else asd_config_path
    expected_config = (
        None
        if config_path is None
        else DSparkASDConfig.from_json(config_path).to_mapping()
    )
    with SGLangServer(
        python=python,
        model_path=model_path,
        artifact_dir=artifact_dir,
        mode=mode,
        config_path=config_path,
        port=port,
    ) as server:
        runner = ProtocolRunner(base_url=server.base_url, model=DEFAULT_MODEL)

        def before_formal() -> None:
            evidence = clear_server_metrics(
                base_url=server.base_url,
                expected_mode=mode,
                expected_config=expected_config,
            )
            write_json(artifact_dir / "pre_formal_clear.json", evidence)

        summary = runner.run_formal_arm(
            warmup_samples=calibration[:10],
            formal_samples=formal,
            warmup_output=artifact_dir / "warmup_outputs.jsonl",
            formal_output=artifact_dir / "formal_outputs.jsonl",
            timing_output=artifact_dir / "formal_timing.json",
            summary_output=artifact_dir / "answer_summary.json",
            acceptance_summary_output=artifact_dir / "acceptance_summary.json",
            before_formal=before_formal,
        )
        counters, trace = collect_server_evidence(
            base_url=server.base_url,
            expected_mode=mode,
            expected_config=expected_config,
            require_calibration_trace=False,
        )
        if trace:
            raise RuntimeError("formal arm unexpectedly emitted calibration trace")
        write_json(artifact_dir / "server_counters.json", counters, immutable=True)
        return summary


def compare_formal_results(
    *, native_summary: Path, asd_summary: Path, output: Path
) -> dict[str, Any]:
    native = json.loads(native_summary.read_text(encoding="utf-8"))
    asd = json.loads(asd_summary.read_text(encoding="utf-8"))

    def metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "completion_tokens": summary["completion_tokens"],
            "timed_wall_seconds": summary["timing"]["timed_wall_seconds"],
            "end_to_end_output_tps": summary["timing"]["end_to_end_output_tps"],
            "gsm8k_matches": summary["matches"],
            "gsm8k_total": summary["total_requests"],
            "failed_requests": summary["failed_requests"],
            "proposal_count": summary["acceptance"]["proposal_count"],
            "accepted_draft_tokens": summary["acceptance"]["accepted_draft_tokens"],
            "mean_accepted_draft_tokens_per_proposal": summary["acceptance"][
                "mean_accepted_draft_tokens_per_proposal"
            ],
            "mean_acceptance_length_including_bonus": summary["acceptance"][
                "mean_acceptance_length_including_bonus"
            ],
        }

    native_metrics = metrics(native)
    asd_metrics = metrics(asd)
    speedup = (
        asd_metrics["end_to_end_output_tps"] / native_metrics["end_to_end_output_tps"]
        - 1
    ) * 100
    comparison = {
        "schema_version": 1,
        "native": native_metrics,
        "asd": asd_metrics,
        "asd_tps_change_percent": speedup,
        "answer_match_change": (
            asd_metrics["gsm8k_matches"] - native_metrics["gsm8k_matches"]
        ),
    }
    write_json(output, comparison)
    return comparison
