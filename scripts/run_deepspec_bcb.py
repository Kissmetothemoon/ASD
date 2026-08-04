#!/usr/bin/env python3
"""Launch a same-GPU DeepSpec DSpark baseline-candidate-baseline triplet.

This wrapper only orchestrates the existing DeepSpec evaluator. It does not
download weights or datasets. Use `--dry-run` to inspect the exact commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def build_command(
    *,
    runner: Path,
    output_dir: Path,
    target: str,
    draft: str,
    dataset_root: Path,
    dataset: str,
    prompts: int,
    offset: int,
    max_new_tokens: int,
    spec_tokens: int,
    seed: int,
    risk_budget: float,
    gate: float,
    cap: int,
) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "--output-dir",
        str(output_dir),
        "--target-name-or-path",
        target,
        "--draft-name-or-path",
        draft,
        "--dataset-root",
        str(dataset_root),
        "--datasets",
        dataset,
        "--num-prompts",
        str(prompts),
        "--dataset-offset",
        str(offset),
        "--dataset-selection",
        "contiguous",
        "--seed",
        str(seed),
        "--num-speculative-tokens",
        str(spec_tokens),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        "0.0",
        "--confidence-threshold",
        "0.0",
        "--horizon-policy",
        "fixed",
        "--risk-budget",
        str(risk_budget),
        "--max-regret-per-value",
        str(gate),
        "--max-relaxed-tokens-per-block",
        str(cap),
        "--reasoning-mode",
        "disable",
        "--tag",
        output_dir.name,
    ]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--spec-tokens", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--risk-budget", type=float, default=8.0)
    parser.add_argument("--gate", type=float, default=0.25)
    parser.add_argument("--cap", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="let the underlying runner handle existing role directories",
    )
    args = parser.parse_args()
    if not args.runner.exists():
        raise FileNotFoundError(args.runner)
    args.output_root.mkdir(parents=True, exist_ok=True)

    roles = (
        ("baseline_pre", 0.0),
        ("candidate", args.risk_budget),
        ("baseline_post", 0.0),
    )
    if not args.dry_run and not args.allow_existing:
        occupied = [
            str(args.output_root / role)
            for role, _ in roles
            if (args.output_root / role).exists()
            and any((args.output_root / role).iterdir())
        ]
        if occupied:
            raise FileExistsError(
                "refusing to reuse non-empty role directories; pass --allow-existing: "
                + ", ".join(occupied)
            )
    runs: list[dict[str, object]] = []
    for role, budget in roles:
        command = build_command(
            runner=args.runner,
            output_dir=args.output_root / role,
            target=args.target,
            draft=args.draft,
            dataset_root=args.dataset_root,
            dataset=args.dataset,
            prompts=args.prompts,
            offset=args.offset,
            max_new_tokens=args.max_new_tokens,
            spec_tokens=args.spec_tokens,
            seed=args.seed,
            risk_budget=budget,
            gate=args.gate,
            cap=args.cap,
        )
        runs.append(
            {
                "role": role,
                "risk_budget": budget,
                "command": command,
            }
        )

    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu

    manifest = {
        "method": "ASD",
        "design": "baseline_pre -> candidate -> baseline_post",
        "target": args.target,
        "draft": args.draft,
        "risk_budget": args.risk_budget,
        "gate": args.gate,
        "cap": args.cap,
        "gpu": args.gpu,
        "runs": runs,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for run in runs:
        command = run["command"]
        assert isinstance(command, list)
        rendered = shlex.join(command)
        print(rendered)
        if not args.dry_run:
            subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
