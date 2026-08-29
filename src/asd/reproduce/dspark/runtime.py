"""Preparation and diagnostics for the pinned DSpark runtime."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import SGLANG_COMMIT, SGLANG_REPOSITORY, TP_SIZE
from .io import write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
INTEGRATION_ROOT = REPOSITORY_ROOT / "integrations" / "sglang-dspark"
PATCH_PATH = INTEGRATION_ROOT / "sglang-0.5.16-asd.patch"
RUNTIME_PROJECT = INTEGRATION_ROOT / "runtime"


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_python(runtime_dir: Path) -> Path:
    return runtime_dir / ".venv" / "bin" / "python"


def _gpu_inventory() -> tuple[list[dict[str, Any]], str | None]:
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi was not found"
    try:
        output = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return [], repr(error)
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            return [], f"unrecognized nvidia-smi row: {line!r}"
        gpus.append(
            {"index": int(parts[0]), "name": parts[1], "memory_mib": int(parts[2])}
        )
    return gpus, None


def doctor(
    *, model_path: Path | None = None, runtime_dir: Path | None = None
) -> dict[str, Any]:
    """Return machine-readable compatibility checks without changing state."""

    gpus, gpu_error = _gpu_inventory()
    model_exists = model_path is not None and model_path.is_dir()
    weight_shards = sorted(model_path.glob("*.safetensors")) if model_exists else []
    checks = {
        "linux": platform.system() == "Linux",
        "uv_available": shutil.which("uv") is not None,
        "git_available": shutil.which("git") is not None,
        "nvidia_smi_available": gpu_error is None,
        "gpu_count_is_8": len(gpus) == TP_SIZE,
        "model_path_exists": model_exists,
        "model_config_exists": bool(
            model_exists and (model_path / "config.json").is_file()
        ),
        "model_weight_shard_count_is_48": len(weight_shards) == 48,
        "runtime_python_exists": (
            runtime_dir is not None and runtime_python(runtime_dir).is_file()
        ),
    }
    required = (
        "linux",
        "uv_available",
        "git_available",
        "nvidia_smi_available",
        "gpu_count_is_8",
        "model_path_exists",
        "model_config_exists",
        "model_weight_shard_count_is_48",
    )
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks[name] for name in required) else "FAIL",
        "checks": checks,
        "gpus": gpus,
        "model_weight_shard_count": len(weight_shards),
        "gpu_inventory_error": gpu_error,
        "note": (
            "The published throughput is hardware-specific (8x NVIDIA H20); "
            "other eight-GPU systems can validate outputs but need not match TPS."
        ),
    }


def prepare_runtime(runtime_dir: Path) -> dict[str, Any]:
    """Clone, verify, patch, resolve, and install the frozen SGLang runtime."""

    if not (INTEGRATION_ROOT / "manifest.json").is_file():
        raise RuntimeError(
            "full DSpark reproduction must be run from an ASD source checkout; "
            "the core-only installed wheel does not contain integration assets"
        )
    uv = shutil.which("uv")
    git = shutil.which("git")
    if uv is None or git is None:
        raise RuntimeError("prepare-runtime requires both git and uv")
    manifest = json.loads((INTEGRATION_ROOT / "manifest.json").read_text())
    expected_patch_hash = manifest["patch"]["sha256"]
    observed_patch_hash = sha256_file(PATCH_PATH)
    if observed_patch_hash != expected_patch_hash:
        raise RuntimeError(f"integration patch hash mismatch: {observed_patch_hash}")
    runtime_files = {
        "pyproject_sha256": RUNTIME_PROJECT / "pyproject.toml",
        "uv_lock_sha256": RUNTIME_PROJECT / "uv.lock",
    }
    for field, path in runtime_files.items():
        observed = sha256_file(path)
        if observed != manifest["runtime"][field]:
            raise RuntimeError(f"runtime manifest hash mismatch for {path}: {observed}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    source = runtime_dir / "sglang"
    if not source.exists():
        _run([git, "clone", SGLANG_REPOSITORY, str(source)])
    if not (source / ".git").is_dir():
        raise RuntimeError(f"runtime source is not a git checkout: {source}")
    status = _run([git, "status", "--porcelain"], cwd=source)
    if status:
        raise RuntimeError(
            "SGLang checkout is dirty; use a fresh --runtime-dir rather than "
            "overwriting local work"
        )
    _run([git, "fetch", "origin", SGLANG_COMMIT], cwd=source)
    _run([git, "checkout", "--detach", SGLANG_COMMIT], cwd=source)
    status = _run([git, "status", "--porcelain"], cwd=source)
    if status:
        raise RuntimeError(
            "SGLang checkout is dirty; use a fresh --runtime-dir rather than "
            "overwriting local work"
        )
    _run(
        [git, "apply", "--check", "--unidiff-zero", str(PATCH_PATH)],
        cwd=source,
    )
    _run([git, "apply", "--unidiff-zero", str(PATCH_PATH)], cwd=source)

    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(runtime_dir / ".venv")
    _run(
        [
            uv,
            "sync",
            "--frozen",
            "--project",
            str(RUNTIME_PROJECT),
            "--python",
            "3.11",
        ],
        environment=environment,
    )
    python = runtime_python(runtime_dir)
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--editable",
            str(REPOSITORY_ROOT),
            "--editable",
            str(source / "python"),
        ]
    )
    identity = {
        "schema_version": 1,
        "status": "PASS",
        "sglang_repository": SGLANG_REPOSITORY,
        "sglang_commit": _run([git, "rev-parse", "HEAD"], cwd=source),
        "patch_sha256": observed_patch_hash,
        "runtime_python": str(python),
        "python_version": _run(
            [str(python), "-c", "import platform; print(platform.python_version())"]
        ),
        "torch_version": _run(
            [str(python), "-c", "import torch; print(torch.__version__)"]
        ),
        "sglang_version": _run(
            [str(python), "-c", "import sglang; print(sglang.__version__)"]
        ),
    }
    write_json(runtime_dir / "runtime_identity.json", identity)
    return identity
