#!/usr/bin/env python3
"""Evaluate three DeepSpec summary CSVs as one paired B-C-B result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from asd.config import ASDConfig
from asd.metrics import evaluate_pair


def read_one(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one summary row in {path}, got {len(rows)}"
        )
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pre", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-post", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workload",
        choices=("fixed", "natural-eos"),
        default="fixed",
        help="fixed requires equal token work; natural-eos evaluates natural lengths",
    )
    args = parser.parse_args()
    config = ASDConfig.from_json(args.config)
    result = evaluate_pair(
        baseline_pre=read_one(args.baseline_pre),
        candidate=read_one(args.candidate),
        baseline_post=read_one(args.baseline_post),
        criteria=config.success_criteria,
        fixed_workload=args.workload == "fixed",
    )
    payload = {
        "method": config.method_name,
        "config": config.__dict__,
        "result": result.__dict__,
        "evidence_status": config.evidence_status,
        "requested_check_passed": result.speed_target_met
        if args.workload == "fixed"
        else result.quality_constraints_met is True,
    }
    rendered = json.dumps(payload, indent=2, default=lambda value: value.__dict__)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["requested_check_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
