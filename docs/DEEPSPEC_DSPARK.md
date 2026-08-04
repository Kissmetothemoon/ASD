# DeepSpec DSpark Integration

This document describes the supported integration point for a greedy DSpark
evaluator. ASD is a verifier-side policy: it runs after the target model has
computed verification logits and before the evaluator commits tokens and crops
its KV cache. It does not train a model or issue an additional target forward
pass.

## Prerequisites

- A DeepSpec/DSpark evaluator that exposes a target-logit tensor for the draft
  positions, with shape `[1, K, V]`.
- Greedy decoding only: `temperature=0`.
- The ASD package installed in the same Python environment as DeepSpec:

```bash
python -m pip install -e '.[torch]'
```

Do not copy DeepSpec into this repository. Read `THIRD_PARTY.md` and obtain a
separately pinned upstream checkout.

## Integration Invariants

1. Create one `RequestRiskState` for each generated request, immediately
   after prefill. Reuse it across every draft block in that request.
2. Pass the target logits for all draft positions, not just the first mismatch.
3. Commit exactly the contiguous prefix returned by ASD, then use the normal
   target greedy bonus-token and KV-cache-cropping path.
4. Keep the target tie rule deterministic. The reference adapter uses PyTorch
   `max`, which selects the first maximum.
5. Set `risk_budget=0` or `max_relaxed_tokens_per_block=0` for the strict
   greedy identity control.

Violating the first or third invariant changes the method and invalidates a
comparison with this reference implementation.

## Minimal Hook

The following belongs in the target verifier after its ordinary greedy prefix
has been computed, and before it chooses the recovery token. `proposal` is a
DeepSpec proposal whose first ID is the current accepted token.

```python
from asd import ASDConfig, RequestRiskState
from asd.adapters import DeepSpecDSparkAdapter

# Create these once per request after prefill, not once per verification block.
asd_config = ASDConfig.from_json("configs/dspark_stable.json")
asd_state = RequestRiskState(total_budget=asd_config.risk_budget)
asd_adapter = DeepSpecDSparkAdapter(asd_config)

# Run after target_output = target_model(...) and after determining K.
draft_token_ids = proposal.verify_input_ids[0, 1 : draft_token_count + 1].tolist()
decision = asd_adapter.decide(
    draft_token_ids=draft_token_ids,
    target_output=target_output.logits[0, :draft_token_count, :],
    state=asd_state,
)
accepted_draft_tokens = decision.accepted_tokens

# Preserve the evaluator's normal greedy recovery and cache logic:
# next_token = target_output.logits[:, accepted_draft_tokens, :].argmax(dim=-1)
```

`patches/deepspec-asd.patch` packages this hook as a reviewable reference
change. It is intentionally small and does not include any DeepSpec file.
Apply it only to the matching evaluator interface, review the resulting diff,
and run the checks below before collecting results.

The research archive used to prepare the patch did not preserve an upstream
Git revision. Its unpatched `deepspec/eval/base_evaluator.py` has SHA-256
`72cac42a41c722746dc923eb91b50d7afab3bf394c2e43ea7cea75e5d95cd8e9`.
If your file differs, use the minimal hook above and record your upstream
commit instead of forcing the patch.

## Request-Scoped State With `generate_decoding_sample`

If the evaluator accepts `verification_kwargs`, pass the state and config to
each call in one request. Construct a fresh dictionary and state for the next
request:

```python
state = RequestRiskState(total_budget=config.risk_budget)
sample = generate_decoding_sample(
    # Existing DeepSpec arguments omitted.
    verification_kwargs={"asd_config": config, "asd_state": state},
)
```

The patch adds these two arguments to `verify_draft_tokens`. A runner that
creates a new state per verification block is not running request-level ASD.

## Smoke Checks

Before a benchmark run, verify all of the following on the same prompts:

```text
1. risk_budget=0 reproduces strict greedy token IDs and completion hashes.
2. The candidate has one RequestRiskState per request, not per block.
3. The candidate uses the existing target bonus token and cache crop path.
4. The evaluator records budget spent, relaxed-token count, and block stopper.
5. A fixed-work run has identical completion-token counts in all B-C-B roles.
```

The package's Python tests validate the selector and protocol checker. They do
not validate a local DeepSpec checkout or certify a model/dataset combination.
The reference adapter also copies compact scores to CPU for clarity. A
performance-oriented integration should implement equivalent GPU-side prefix
selection while retaining the same strict-identity and request-ledger tests.
