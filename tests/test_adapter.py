import unittest

try:
    import torch
except ImportError:
    torch = None

from asd.adapters.deepspec import DeepSpecDSparkAdapter, FullLogitScoreProvider
from asd.budget import RequestRiskState, TokenScores
from asd.config import ASDConfig


class MockProvider:
    def score_draft(self, draft_token_ids, target_output):
        del draft_token_ids, target_output
        return TokenScores((5.0, 5.0), (1, 2), (4.9, 5.0))


class AdapterTests(unittest.TestCase):
    def test_deepspec_adapter_accepts_compact_score_provider(self):
        config = ASDConfig(1.0, 0.25, 1)
        adapter = DeepSpecDSparkAdapter(config=config, score_provider=MockProvider())
        decision = adapter.decide(
            draft_token_ids=[9, 2],
            target_output=object(),
            state=RequestRiskState(1.0),
        )
        self.assertEqual(decision.accepted_tokens, 2)
        self.assertEqual(decision.relaxed_tokens, 1)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_full_logit_provider_ignores_recovery_row(self):
        provider = FullLogitScoreProvider()
        logits = torch.tensor(
            [[[5.0, 4.9], [1.0, 3.0], [9.0, 0.0]]],
            dtype=torch.float32,
        )
        scores = provider.score_draft([1, 1], logits)
        self.assertEqual(scores.top_token_ids, (0, 1))
        self.assertAlmostEqual(scores.draft_logits[0], 4.9, places=6)
        self.assertAlmostEqual(scores.draft_logits[1], 3.0)


if __name__ == "__main__":
    unittest.main()
