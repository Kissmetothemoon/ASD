"""Small command-line entry point for reference demos and pair checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .budget import RequestRiskState, TokenScores, choose_prefix
from .config import ASDConfig
from .metrics import evaluate_pair


def demo() -> int:
    config = ASDConfig(
        risk_budget=1.0,
        max_regret_per_value=0.25,
        max_relaxed_tokens_per_block=1,
    )
    state = RequestRiskState(total_budget=config.risk_budget)
    scores = TokenScores(
        top_logits=(5.0, 4.0, 3.0),
        top_token_ids=(10, 11, 12),
        draft_logits=(4.8, 4.0, 2.0),
    )
    decision = choose_prefix(
        draft_token_ids=(9, 11, 13),
        scores=scores,
        state=state,
        config=config,
    )
    print(json.dumps({"decision": decision.__dict__, "state": state.__dict__}, indent=2))
    return 0


def pair_check(path: str) -> int:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    criteria = ASDConfig.from_mapping(payload.get("config", {})).success_criteria
    result = evaluate_pair(
        baseline_pre=payload["baseline_pre"],
        candidate=payload["candidate"],
        baseline_post=payload["baseline_post"],
        criteria=criteria,
        fixed_workload=bool(payload.get("fixed_workload", True)),
    )
    print(json.dumps(result.__dict__, indent=2))
    return 0 if result.overall_success else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="ASD reference utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run a dependency-free prefix-selection demo")
    pair = sub.add_parser("pair-check", help="evaluate one JSON B-C-B pair")
    pair.add_argument("path")
    args = parser.parse_args()
    if args.command == "demo":
        return demo()
    return pair_check(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
