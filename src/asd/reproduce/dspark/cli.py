"""Command-line entry point for the frozen DSpark reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import DATASET_MANIFEST_NAME, verify_materialized_dataset
from .experiment import (
    compare_formal_results,
    download_model,
    prepare_data,
    reduce_calibration_run,
    run_calibration_arm,
    run_formal_arm,
)
from .runtime import doctor, prepare_runtime, runtime_python


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _require_absent_or_complete(
    directory: Path, completion_file: str, *, resume: bool
) -> bool:
    complete = directory / completion_file
    if resume and complete.is_file():
        return False
    if directory.exists():
        raise FileExistsError(
            f"refusing to overwrite incomplete/existing stage {directory}; "
            "choose a new output directory"
        )
    return True


def _add_execution_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=_path, required=True)
    parser.add_argument(
        "--runtime-dir", type=_path, default=_path(".asd-runtime/dspark")
    )
    parser.add_argument(
        "--output-dir", type=_path, default=_path("runs/dspark-reproduction")
    )
    parser.add_argument("--port", type=int, default=31066)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asd-dspark-reproduce",
        description=(
            "Reproduce ASD on DeepSeek-V4-Flash-DSpark with the frozen GSM8K protocol."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("doctor", help="inspect host compatibility")
    check.add_argument("--model-path", type=_path)
    check.add_argument("--runtime-dir", type=_path)

    model = subparsers.add_parser(
        "download-model", help="download the exact Hugging Face model revision"
    )
    model.add_argument("--output-dir", type=_path, required=True)

    runtime = subparsers.add_parser(
        "prepare-runtime", help="build the pinned patched SGLang environment"
    )
    runtime.add_argument("--runtime-dir", type=_path, required=True)

    data = subparsers.add_parser(
        "prepare-data", help="materialize the pinned GSM8K cohorts"
    )
    data.add_argument("--output-dir", type=_path, required=True)

    calibration = subparsers.add_parser(
        "run-calibration", help="run one of the two 32-request calibration arms"
    )
    _add_execution_paths(calibration)
    calibration.add_argument("--arm", choices=("native-trace", "b0"), required=True)

    reduce = subparsers.add_parser(
        "reduce-calibration", help="prove B=0 equivalence and derive q25"
    )
    reduce.add_argument("--output-dir", type=_path, required=True)

    formal = subparsers.add_parser(
        "run-formal", help="run one warmup-10 plus timed formal-500 arm"
    )
    _add_execution_paths(formal)
    formal.add_argument("--arm", choices=("native", "asd"), required=True)

    compare = subparsers.add_parser(
        "compare", help="recompute the native-versus-ASD result table"
    )
    compare.add_argument("--output-dir", type=_path, required=True)

    all_parser = subparsers.add_parser(
        "all", help="run preparation, calibration, and both formal arms"
    )
    _add_execution_paths(all_parser)
    all_parser.add_argument("--resume", action="store_true")
    all_parser.add_argument(
        "--dry-run", action="store_true", help="print stages without changing state"
    )
    return parser


def _run_calibration_command(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir
    data_dir = root / "data"
    artifact_dir = root / "calibration" / args.arm
    return run_calibration_arm(
        arm=args.arm,
        python=runtime_python(args.runtime_dir),
        model_path=args.model_path,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        port=args.port,
    )


def _run_formal_command(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir
    return run_formal_arm(
        arm=args.arm,
        python=runtime_python(args.runtime_dir),
        model_path=args.model_path,
        data_dir=root / "data",
        artifact_dir=root / "formal" / args.arm,
        asd_config_path=root / "calibration" / "reduced" / "asd_config.json",
        port=args.port,
    )


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir
    plan = {
        "model_path": str(args.model_path),
        "runtime_dir": str(args.runtime_dir),
        "output_dir": str(root),
        "port": args.port,
        "stages": [
            "doctor",
            "prepare-runtime",
            "prepare-data",
            "calibration/native-trace (32)",
            "calibration/b0 (32)",
            "reduce-calibration and verify committed q25 config",
            "formal/native (warmup 10 + timed 500)",
            "formal/asd (warmup 10 + timed 500)",
            "compare",
        ],
        "server_restarts": 4,
    }
    if args.dry_run:
        return {"schema_version": 1, "status": "DRY_RUN", **plan}

    report = doctor(model_path=args.model_path, runtime_dir=args.runtime_dir)
    if report["status"] != "PASS":
        raise RuntimeError(f"host compatibility checks failed: {report['checks']}")
    root.mkdir(parents=True, exist_ok=True)

    runtime_identity = args.runtime_dir / "runtime_identity.json"
    if not (args.resume and runtime_identity.is_file()):
        prepare_runtime(args.runtime_dir)

    data_dir = root / "data"
    if args.resume and (data_dir / DATASET_MANIFEST_NAME).is_file():
        verify_materialized_dataset(data_dir)
    elif data_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing data stage: {data_dir}")
    else:
        prepare_data(data_dir)

    for arm in ("native-trace", "b0"):
        directory = root / "calibration" / arm
        if _require_absent_or_complete(directory, "summary.json", resume=args.resume):
            run_calibration_arm(
                arm=arm,
                python=runtime_python(args.runtime_dir),
                model_path=args.model_path,
                data_dir=data_dir,
                artifact_dir=directory,
                port=args.port,
            )

    reduced = root / "calibration" / "reduced"
    if _require_absent_or_complete(
        reduced, "calibration_summary.json", resume=args.resume
    ):
        reduce_calibration_run(root / "calibration")

    for arm in ("native", "asd"):
        directory = root / "formal" / arm
        if _require_absent_or_complete(
            directory, "answer_summary.json", resume=args.resume
        ):
            run_formal_arm(
                arm=arm,
                python=runtime_python(args.runtime_dir),
                model_path=args.model_path,
                data_dir=data_dir,
                artifact_dir=directory,
                asd_config_path=reduced / "asd_config.json",
                port=args.port,
            )
    comparison = compare_formal_results(
        native_summary=root / "formal" / "native" / "answer_summary.json",
        asd_summary=root / "formal" / "asd" / "answer_summary.json",
        output=root / "comparison.json",
    )
    return {"schema_version": 1, "status": "PASS", **plan, "comparison": comparison}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = doctor(model_path=args.model_path, runtime_dir=args.runtime_dir)
        _emit(result)
        return 0 if result["status"] == "PASS" else 1
    if args.command == "download-model":
        _emit({"model_path": str(download_model(args.output_dir))})
        return 0
    if args.command == "prepare-runtime":
        _emit(prepare_runtime(args.runtime_dir))
        return 0
    if args.command == "prepare-data":
        _emit(prepare_data(args.output_dir))
        return 0
    if args.command == "run-calibration":
        _emit(_run_calibration_command(args))
        return 0
    if args.command == "reduce-calibration":
        _emit(reduce_calibration_run(args.output_dir / "calibration"))
        return 0
    if args.command == "run-formal":
        _emit(_run_formal_command(args))
        return 0
    if args.command == "compare":
        root = args.output_dir
        _emit(
            compare_formal_results(
                native_summary=root / "formal" / "native" / "answer_summary.json",
                asd_summary=root / "formal" / "asd" / "answer_summary.json",
                output=root / "comparison.json",
            )
        )
        return 0
    if args.command == "all":
        _emit(run_all(args))
        return 0
    parser.error(f"unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
