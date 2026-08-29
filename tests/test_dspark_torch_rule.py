"""Fixed-seed parity tests for the device-resident tensor rule."""

from __future__ import annotations

import random
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from asd.reproduce.dspark.asd_config import DSparkASDConfig
from asd.reproduce.dspark.rule import (
    RequestRiskState,
    TokenScores,
    choose_prefix,
)

if torch is not None:
    from asd.reproduce.dspark.torch_rule import choose_prefix_batch


def config(*, B: float, g: float, m: int, block_size: int = 5) -> DSparkASDConfig:
    return DSparkASDConfig(
        risk_budget=B,
        max_regret_per_value=g,
        max_relaxed_mismatches_per_block=m,
        block_size=block_size,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TensorRuleTests(unittest.TestCase):
    def test_fixed_seed_width_five_blocks_match_reference(self) -> None:
        rng = random.Random(20260725)
        budgets = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0)
        gates = (0.0, 0.1, 0.25, 0.5, 2.0)
        gaps = (0.0, 0.05, 0.125, 0.25, 0.5, 1.0)

        for trial in range(512):
            B = rng.choice(budgets)
            cfg = config(B=B, g=rng.choice(gates), m=rng.randint(0, 5))
            already_spent = 0.0 if B == 0.0 else rng.choice((0.0, B / 2, B))
            state = RequestRiskState(
                total_budget=B,
                spent=already_spent,
            )
            top_ids = tuple(rng.randrange(0, 32) for _ in range(5))
            draft_ids = tuple(
                top_id if rng.random() < 0.55 else top_id + 64 for top_id in top_ids
            )
            top_logits = tuple(rng.uniform(-4.0, 4.0) for _ in range(5))
            chosen_gaps = tuple(rng.choice(gaps) for _ in range(5))
            draft_logits = tuple(
                top_logit - gap for top_logit, gap in zip(top_logits, chosen_gaps)
            )

            expected = choose_prefix(
                draft_token_ids=draft_ids,
                scores=TokenScores(
                    top_logits=top_logits,
                    top_token_ids=top_ids,
                    draft_logits=draft_logits,
                ),
                state=state,
                config=cfg,
            )
            actual = choose_prefix_batch(
                draft_token_ids=torch.tensor(
                    [draft_ids],
                    dtype=torch.int64,
                ),
                top_logits=torch.tensor(
                    [top_logits],
                    dtype=torch.float64,
                ),
                top_token_ids=torch.tensor(
                    [top_ids],
                    dtype=torch.int64,
                ),
                draft_logits=torch.tensor(
                    [draft_logits],
                    dtype=torch.float64,
                ),
                remaining_budget=torch.tensor(
                    [B - already_spent],
                    dtype=torch.float64,
                ),
                config=cfg,
            )

            context = (
                f"trial={trial}, B={B}, spent={already_spent}, g={cfg.g}, m={cfg.m}"
            )
            self.assertEqual(
                int(actual.accepted[0]),
                expected.accepted_tokens,
                context,
            )
            self.assertEqual(
                int(actual.relaxed_mismatches[0]),
                expected.relaxed_mismatches,
                context,
            )
            self.assertAlmostEqual(
                float(actual.spent[0]),
                expected.risk_spent,
                places=12,
                msg=context,
            )
            self.assertEqual(
                tuple(bool(value) for value in actual.mismatched[0]),
                expected.mismatched,
                context,
            )
            for position in range(5):
                self.assertAlmostEqual(
                    float(actual.regrets[0, position]),
                    expected.token_regrets[position],
                    places=12,
                    msg=context,
                )
                self.assertAlmostEqual(
                    float(actual.values[0, position]),
                    expected.token_values[position],
                    places=12,
                    msg=context,
                )

    def test_width_five_tensor_values_are_frozen(self) -> None:
        decision = choose_prefix_batch(
            draft_token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            top_logits=torch.ones((1, 5), dtype=torch.float64),
            top_token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            draft_logits=torch.ones((1, 5), dtype=torch.float64),
            remaining_budget=torch.tensor([0.0], dtype=torch.float64),
            config=config(B=0.0, g=0.0, m=0),
        )
        self.assertEqual(
            tuple(float(value) for value in decision.values[0]),
            (1.0, 0.8, 0.6, 0.4, 0.2),
        )

    def test_B_zero_and_zero_regret_mismatch_are_strict(self) -> None:
        decision = choose_prefix_batch(
            draft_token_ids=torch.tensor([[1, 2, 9, 4, 5]]),
            top_logits=torch.ones((1, 5), dtype=torch.float64),
            top_token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            # The mismatch at position two is a target-logit tie.
            draft_logits=torch.ones((1, 5), dtype=torch.float64),
            remaining_budget=torch.tensor([0.0], dtype=torch.float64),
            config=config(B=0.0, g=1_000_000.0, m=5),
        )
        self.assertEqual(int(decision.accepted[0]), 2)
        self.assertEqual(int(decision.relaxed_mismatches[0]), 0)
        self.assertEqual(float(decision.spent[0]), 0.0)

    def test_contiguous_prefix_blocks_later_exact_positions(self) -> None:
        decision = choose_prefix_batch(
            draft_token_ids=torch.tensor([[9, 2, 3, 4, 5]]),
            top_logits=torch.ones((1, 5), dtype=torch.float64),
            top_token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            draft_logits=torch.tensor(
                [[0.0, 1.0, 1.0, 1.0, 1.0]],
                dtype=torch.float64,
            ),
            remaining_budget=torch.tensor([10.0], dtype=torch.float64),
            config=config(B=10.0, g=0.01, m=5),
        )
        self.assertEqual(int(decision.accepted[0]), 0)

    def test_m_one_caps_the_second_mismatch(self) -> None:
        decision = choose_prefix_batch(
            draft_token_ids=torch.tensor([[9, 2, 8, 4, 5]]),
            top_logits=torch.ones((1, 5), dtype=torch.float64),
            top_token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            draft_logits=torch.tensor(
                [[0.9, 1.0, 0.9, 1.0, 1.0]],
                dtype=torch.float64,
            ),
            remaining_budget=torch.tensor([10.0], dtype=torch.float64),
            config=config(B=10.0, g=10.0, m=1),
        )
        self.assertEqual(int(decision.accepted[0]), 2)
        self.assertEqual(int(decision.relaxed_mismatches[0]), 1)

    def test_rows_have_independent_remaining_budgets(self) -> None:
        decision = choose_prefix_batch(
            draft_token_ids=torch.tensor(
                [[9, 8, 7, 6, 5], [9, 8, 7, 6, 5]],
            ),
            top_logits=torch.ones((2, 5), dtype=torch.float64),
            top_token_ids=torch.tensor(
                [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]],
            ),
            draft_logits=torch.full((2, 5), 0.5, dtype=torch.float64),
            remaining_budget=torch.tensor([0.4, 1.0], dtype=torch.float64),
            config=config(B=1.0, g=10.0, m=5),
        )
        self.assertEqual(tuple(int(value) for value in decision.accepted), (0, 2))
        self.assertEqual(
            tuple(int(value) for value in decision.relaxed_mismatches),
            (0, 2),
        )

    def test_budget_spent_tensor_persists_across_blocks(self) -> None:
        cfg = config(B=0.25, g=10.0, m=2)
        remaining = torch.tensor([cfg.B], dtype=torch.float64)
        first = choose_prefix_batch(
            draft_token_ids=torch.tensor([[9, 2, 3, 4, 5]]),
            top_logits=torch.ones((1, 5), dtype=torch.float64),
            top_token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            draft_logits=torch.tensor(
                [[0.75, 1.0, 1.0, 1.0, 1.0]],
                dtype=torch.float64,
            ),
            remaining_budget=remaining,
            config=cfg,
        )
        remaining = remaining - first.spent
        second = choose_prefix_batch(
            draft_token_ids=torch.tensor([[1, 9, 3, 4, 5]]),
            top_logits=torch.ones((1, 5), dtype=torch.float64),
            top_token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            draft_logits=torch.ones((1, 5), dtype=torch.float64),
            remaining_budget=remaining,
            config=cfg,
        )
        self.assertEqual(float(remaining[0]), 0.0)
        self.assertEqual(int(second.accepted[0]), 1)
        self.assertEqual(int(second.relaxed_mismatches[0]), 0)

    def test_exact_exhaustion_inside_block_blocks_later_tie(self) -> None:
        decision = choose_prefix_batch(
            draft_token_ids=torch.tensor([[9, 2, 8, 4, 5]]),
            top_logits=torch.ones((1, 5), dtype=torch.float64),
            top_token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            draft_logits=torch.tensor(
                [[0.75, 1.0, 1.0, 1.0, 1.0]],
                dtype=torch.float64,
            ),
            remaining_budget=torch.tensor([0.25], dtype=torch.float64),
            config=config(B=0.25, g=10.0, m=2),
        )
        self.assertEqual(int(decision.accepted[0]), 2)
        self.assertEqual(int(decision.relaxed_mismatches[0]), 1)

    def test_block_size_mismatch_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "observed block_size=4"):
            choose_prefix_batch(
                draft_token_ids=torch.tensor([[1, 2, 3, 4]]),
                top_logits=torch.ones((1, 4), dtype=torch.float64),
                top_token_ids=torch.tensor([[1, 2, 3, 4]]),
                draft_logits=torch.ones((1, 4), dtype=torch.float64),
                remaining_budget=torch.tensor([0.0], dtype=torch.float64),
                config=config(B=0.0, g=0.0, m=0),
            )

    def test_hot_rule_does_not_call_tensor_item(self) -> None:
        args = {
            "draft_token_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "top_logits": torch.ones((1, 5), dtype=torch.float64),
            "top_token_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "draft_logits": torch.ones((1, 5), dtype=torch.float64),
            "remaining_budget": torch.tensor([0.0], dtype=torch.float64),
            "config": config(B=0.0, g=0.0, m=0),
        }
        with mock.patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("host sync via .item()"),
        ):
            decision = choose_prefix_batch(**args)
        self.assertEqual(tuple(decision.accepted.shape), (1,))


if __name__ == "__main__":
    unittest.main()
