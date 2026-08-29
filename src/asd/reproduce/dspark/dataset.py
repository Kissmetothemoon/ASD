"""Pinned GSM8K acquisition and deterministic cohort materialization."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .answers import extract_reference_answer
from .config import (
    CALIBRATION_COUNT,
    DATASET_CONFIG,
    DATASET_PROVIDER,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    FORMAL_COUNT,
    SHUFFLE_SEED,
    WARMUP_COUNT,
    build_user_content,
    canonical_json_bytes,
    protocol_config_document,
    sha256_bytes,
)
from .io import load_jsonl, sha256_file, write_immutable, write_json

DATASET_MANIFEST_NAME = "dataset_manifest.json"
CALIBRATION_JSONL_NAME = "gsm8k_calibration_32.jsonl"
FORMAL_JSONL_NAME = "gsm8k_formal_500.jsonl"
PROTOCOL_CONFIG_NAME = "protocol_config.json"


def selected_indices(test_size: int) -> tuple[list[int], list[int]]:
    if test_size < CALIBRATION_COUNT + FORMAL_COUNT:
        raise ValueError(
            f"dataset has {test_size} rows; need at least "
            f"{CALIBRATION_COUNT + FORMAL_COUNT}"
        )
    indices = list(range(test_size))
    random.Random(SHUFFLE_SEED).shuffle(indices)
    calibration = indices[:CALIBRATION_COUNT]
    formal = indices[CALIBRATION_COUNT : CALIBRATION_COUNT + FORMAL_COUNT]
    if set(calibration) & set(formal):
        raise AssertionError("calibration/formal split unexpectedly overlaps")
    return calibration, formal


def _record(
    *,
    cohort: str,
    cohort_position: int,
    dataset_index: int,
    row: Mapping[str, Any],
    dataset_fingerprint: str,
) -> dict[str, Any]:
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise TypeError(
            f"GSM8K row {dataset_index} must contain string question/answer"
        )
    reference = extract_reference_answer(answer)
    if reference.value is None:
        raise ValueError(
            f"GSM8K row {dataset_index} has no parseable final #### answer"
        )
    return {
        "schema_version": 1,
        "cohort": cohort,
        "cohort_position": cohort_position,
        "dataset_index": dataset_index,
        "dataset_revision": DATASET_REVISION,
        "dataset_fingerprint": dataset_fingerprint,
        "question": question,
        "answer": answer,
        "reference_answer": reference.value,
        "reference_answer_raw": reference.raw,
        "reference_extraction_rule": reference.rule,
        "user_content": build_user_content(question),
    }


def _rows_content_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        digest.update(
            canonical_json_bytes(
                {
                    "dataset_index": index,
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                }
            )
        )
    return digest.hexdigest()


def _git_head(repository_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def materialize_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    resolved_revision: str,
    dataset_fingerprint: str,
    output_dir: Path,
    generator_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Write immutable 32/500 cohorts and their provenance manifest."""

    if resolved_revision != DATASET_REVISION:
        raise ValueError(
            f"resolved revision {resolved_revision!r} != pinned {DATASET_REVISION!r}"
        )
    if not dataset_fingerprint:
        raise ValueError("dataset_fingerprint must be non-empty")
    calibration_indices, formal_indices = selected_indices(len(rows))
    calibration_records = [
        _record(
            cohort="calibration",
            cohort_position=position,
            dataset_index=index,
            row=rows[index],
            dataset_fingerprint=dataset_fingerprint,
        )
        for position, index in enumerate(calibration_indices)
    ]
    formal_records = [
        _record(
            cohort="formal",
            cohort_position=position,
            dataset_index=index,
            row=rows[index],
            dataset_fingerprint=dataset_fingerprint,
        )
        for position, index in enumerate(formal_indices)
    ]
    calibration_payload = b"".join(
        canonical_json_bytes(record) for record in calibration_records
    )
    formal_payload = b"".join(canonical_json_bytes(record) for record in formal_records)
    protocol_config = protocol_config_document()
    protocol_payload = (
        json.dumps(
            protocol_config,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / CALIBRATION_JSONL_NAME
    formal_path = output_dir / FORMAL_JSONL_NAME
    protocol_path = output_dir / PROTOCOL_CONFIG_NAME
    write_immutable(calibration_path, calibration_payload)
    write_immutable(formal_path, formal_payload)
    write_immutable(protocol_path, protocol_payload)

    shuffled_indices = list(range(len(rows)))
    random.Random(SHUFFLE_SEED).shuffle(shuffled_indices)
    generator_bytes = generator_path.read_bytes()
    generator_files: dict[str, str] = {}
    candidate_generator_files = [
        generator_path,
        *sorted(Path(__file__).resolve().parent.glob("*.py")),
    ]
    for candidate in candidate_generator_files:
        try:
            relative = candidate.resolve().relative_to(repository_root.resolve())
        except ValueError:
            continue
        generator_files[str(relative)] = sha256_bytes(candidate.read_bytes())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "immutable": True,
        "source": {
            "provider": DATASET_PROVIDER,
            "repo_id": DATASET_REPO_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "resolved_revision": resolved_revision,
        },
        "dataset": {
            "row_count": len(rows),
            "column_names": ["question", "answer"],
            "datasets_fingerprint": dataset_fingerprint,
            "content_sha256": _rows_content_sha256(rows),
        },
        "selection": {
            "algorithm": (
                "indices=list(range(test_size)); random.Random(980406).shuffle(indices)"
            ),
            "seed": SHUFFLE_SEED,
            "shuffled_indices_sha256": sha256_bytes(
                canonical_json_bytes(shuffled_indices)
            ),
            "calibration_count": CALIBRATION_COUNT,
            "calibration_indices": calibration_indices,
            "formal_count": FORMAL_COUNT,
            "formal_indices": formal_indices,
            "warmup_count": WARMUP_COUNT,
            "warmup_indices": calibration_indices[:WARMUP_COUNT],
            "cohorts_disjoint": not bool(
                set(calibration_indices) & set(formal_indices)
            ),
        },
        "artifacts": {
            CALIBRATION_JSONL_NAME: {
                "row_count": len(calibration_records),
                "sha256": sha256_bytes(calibration_payload),
            },
            FORMAL_JSONL_NAME: {
                "row_count": len(formal_records),
                "sha256": sha256_bytes(formal_payload),
            },
            PROTOCOL_CONFIG_NAME: {
                "fingerprint_sha256": protocol_config["fingerprint_sha256"],
                "sha256": sha256_bytes(protocol_payload),
            },
        },
        "generator": {
            "path": str(generator_path.relative_to(repository_root)),
            "source_sha256": sha256_bytes(generator_bytes),
            "files": generator_files,
            # The generator cannot embed the hash of the commit that contains
            # this manifest without a self-reference. The enclosing P01 commit
            # is authoritative; this records the exact base plus source bytes.
            "repository_head_at_generation": _git_head(repository_root),
        },
    }
    manifest_path = output_dir / DATASET_MANIFEST_NAME
    write_json(manifest_path, manifest, immutable=True)

    # Direct post-write verification catches encoding or filesystem surprises.
    if (
        sha256_file(calibration_path)
        != manifest["artifacts"][CALIBRATION_JSONL_NAME]["sha256"]
    ):
        raise RuntimeError("calibration JSONL hash changed after write")
    if sha256_file(formal_path) != manifest["artifacts"][FORMAL_JSONL_NAME]["sha256"]:
        raise RuntimeError("formal JSONL hash changed after write")
    return manifest


def verify_materialized_dataset(
    output_dir: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Recompute the locally provable immutable-manifest invariants."""

    manifest_path = output_dir / DATASET_MANIFEST_NAME
    calibration_path = output_dir / CALIBRATION_JSONL_NAME
    formal_path = output_dir / FORMAL_JSONL_NAME
    protocol_path = output_dir / PROTOCOL_CONFIG_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    calibration = load_jsonl(calibration_path)
    formal = load_jsonl(formal_path)
    checks: dict[str, bool] = {
        "pinned_revision": (
            manifest["source"]["revision"] == DATASET_REVISION
            and manifest["source"]["resolved_revision"] == DATASET_REVISION
        ),
        "protocol_fingerprint": (
            protocol["fingerprint_sha256"]
            == protocol_config_document()["fingerprint_sha256"]
            and protocol == protocol_config_document()
        ),
        "calibration_count": len(calibration) == CALIBRATION_COUNT,
        "formal_count": len(formal) == FORMAL_COUNT,
        "calibration_hash": (
            sha256_file(calibration_path)
            == manifest["artifacts"][CALIBRATION_JSONL_NAME]["sha256"]
        ),
        "formal_hash": (
            sha256_file(formal_path)
            == manifest["artifacts"][FORMAL_JSONL_NAME]["sha256"]
        ),
        "protocol_hash": (
            sha256_file(protocol_path)
            == manifest["artifacts"][PROTOCOL_CONFIG_NAME]["sha256"]
        ),
    }
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[4]
    generator_files = manifest.get("generator", {}).get("files", {})
    expected_generator_files: set[str] = set()
    candidate_generator_files = [
        repository_root / manifest["generator"]["path"],
        *sorted(Path(__file__).resolve().parent.glob("*.py")),
    ]
    for candidate in candidate_generator_files:
        try:
            relative = candidate.resolve().relative_to(repository_root.resolve())
        except ValueError:
            continue
        expected_generator_files.add(str(relative))
    checks["generator_file_set"] = set(generator_files) == expected_generator_files
    checks["generator_sources"] = bool(generator_files) and all(
        (repository_root / relative).is_file()
        and sha256_file(repository_root / relative) == expected_hash
        for relative, expected_hash in generator_files.items()
    )
    expected_calibration, expected_formal = selected_indices(
        manifest["dataset"]["row_count"]
    )
    calibration_indices = [row["dataset_index"] for row in calibration]
    formal_indices = [row["dataset_index"] for row in formal]
    checks.update(
        calibration_indices=calibration_indices == expected_calibration,
        formal_indices=formal_indices == expected_formal,
        manifest_calibration_indices=(
            manifest["selection"]["calibration_indices"] == calibration_indices
        ),
        manifest_formal_indices=(
            manifest["selection"]["formal_indices"] == formal_indices
        ),
        cohorts_disjoint=not bool(set(calibration_indices) & set(formal_indices)),
        warmup_indices=(
            manifest["selection"]["warmup_indices"]
            == calibration_indices[:WARMUP_COUNT]
        ),
    )
    all_records = [*calibration, *formal]
    checks["record_identity"] = all(
        row["dataset_revision"] == DATASET_REVISION
        and row["dataset_fingerprint"] == manifest["dataset"]["datasets_fingerprint"]
        for row in all_records
    )
    checks["prompt_exact"] = all(
        row["user_content"] == build_user_content(row["question"])
        for row in all_records
    )
    checks["reference_recomputes"] = all(
        extract_reference_answer(row["answer"]).value == row["reference_answer"]
        for row in all_records
    )
    result = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source": manifest["source"],
        "dataset_fingerprint": manifest["dataset"]["datasets_fingerprint"],
        "artifact_hashes": {
            CALIBRATION_JSONL_NAME: sha256_file(calibration_path),
            FORMAL_JSONL_NAME: sha256_file(formal_path),
            PROTOCOL_CONFIG_NAME: sha256_file(protocol_path),
        },
    }
    if result["status"] != "PASS":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"materialized dataset verification failed: {failed}")
    return result


def acquire_and_materialize(
    *,
    output_dir: Path,
    cache_dir: Path,
    generator_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Acquire only the pinned HF revision and materialize protocol artifacts."""

    from datasets import load_dataset
    from huggingface_hub import HfApi

    resolved = HfApi().dataset_info(DATASET_REPO_ID, revision=DATASET_REVISION).sha
    if resolved != DATASET_REVISION:
        raise RuntimeError(
            f"Hugging Face resolved {resolved!r}, expected {DATASET_REVISION!r}"
        )
    dataset = load_dataset(
        DATASET_REPO_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        cache_dir=str(cache_dir),
    )
    if list(dataset.column_names) != ["question", "answer"]:
        raise RuntimeError(f"unexpected GSM8K columns: {list(dataset.column_names)!r}")
    rows = [{"question": row["question"], "answer": row["answer"]} for row in dataset]
    return materialize_dataset(
        rows,
        resolved_revision=resolved,
        dataset_fingerprint=dataset._fingerprint,
        output_dir=output_dir,
        generator_path=generator_path,
        repository_root=repository_root,
    )
