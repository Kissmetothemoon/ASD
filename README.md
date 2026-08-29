# Approximate Speculative Decoding

Reference code and a DeepSpec/DSpark integration case for **Approximate
Speculative Decoding (ASD)**. ASD is a training-free, verifier-side policy for
greedy speculative decoding. It can accept a small number of non-greedy draft
tokens under explicit token-, block-, and request-level controls, allowing an
already verified contiguous suffix to be committed instead of discarded.

> **ASD is approximate decoding.** It does not preserve the target model's
> greedy output or sampling distribution. Its risk ledger bounds realized
> local target-logit regret; it does not guarantee task accuracy, semantic
> quality, safety, or output identity.

## Release Scope

This repository keeps the smaller Qwen experiments as the primary,
easier-to-run ASD examples:

- a dependency-free Python implementation of ASD prefix selection;
- a PyTorch full-logit adapter for the DeepSpec DSpark verifier;
- a minimal patch and integration guide for an upstream DeepSpec checkout;
- frozen and discovery-only Qwen/DSpark configurations;
- a same-GPU baseline-candidate-baseline (B-C-B) launcher and result checker;
- CPU-only unit tests and GitHub Actions CI.

It additionally contains a separate, pinned reproduction package for the
DeepSeek-V4-Flash-DSpark experiment. That 8-GPU experiment is a large-model
showcase, not a replacement for the Qwen workflow or its ASD configuration.

DeepSpec, SGLang, model weights, DSpark checkpoints, datasets, benchmark
answers, and generated experiment outputs are not included. See
`THIRD_PARTY.md`.

## Patent and IP

An ASD-related Chinese patent application was filed before the first planned
public release of this repository. The public repository intentionally does
not disclose its application number, filing date, unpublished claim set, or
unfiled improvements.

This repository is licensed under Apache License 2.0. Its Section 3 supplies
the patent license for patent claims that a contributor can license and that
are necessarily infringed by that contributor's released contribution, subject
to the license terms. No separate or broader patent license is implied.
`PATENT_AND_IP.md` defines the public scope and contribution expectations;
`RELEASE_CHECKLIST.md` records the controls required before a public release.
Working paper PDFs are excluded from the source release by default and require
separate venue and copyright approval before they are explicitly published.

## How ASD Works

For draft token `x_i` and target logits `z_i`, ASD charges the local regret

```text
r_i = max_v z_i(v) - z_i(x_i).
```

An exact target-argmax token costs zero. A mismatched token is accepted only if
all of the following remain true for the contiguous draft prefix:

1. request-wide cumulative regret does not exceed budget `B`;
2. local regret per suffix-value unit satisfies `r_i / q_i <= g`;
3. the block contains no more than `m` accepted mismatches.

The reference suffix value is `q_i = K - i` in zero-based Python indexing.
The first infeasible position stops the draft prefix. The normal target greedy
bonus token is then committed. Setting `B=0` or `m=0` recovers strict greedy
verification.

## Reported DSpark Effect

The following is an experiment snapshot, not a universal performance claim.
It uses the frozen configuration `B=8`, `g=0.25`, `m=2`, greedy decoding, and
unmodified target/draft checkpoints.

| Setting | Measured result |
| --- | --- |
| Qwen3-8B + matched DSpark-8B, GSM8K, 64 prompts x 256 fixed tokens | `+4.53%` TPS over strict DSpark across 9 B-C-B runs; 95% CI `[+4.13%, +4.92%]` |
| Qwen3-14B + matched DSpark-14B, 7 fixed-work tasks | Positive on all tasks; unweighted mean `+7.90%` over strict DSpark |
| Qwen3-14B per-task range | `+3.86%` to `+11.74%`, 4 B-C-B repetitions per task |
| Hardware/runtime used for these campaigns | One NVIDIA L20 per triplet; Python 3.12.13, PyTorch 2.8.0+cu128, Transformers 4.55.2 |

Speed and quality were measured separately. Natural-EOS audits showed that
outputs can change substantially: Qwen3-14B + DSpark-14B had a `-0.61`
percentage-point HumanEval change in the reported audit, while GSM8K and
MATH-500 completion-hash divergence exceeded 95%. These observations are why
the code reports speed eligibility and quality constraints separately. Re-run
the B-C-B and natural-EOS protocols on every deployment workload.

### DeepSeek large-model showcase

The repository also records one historical run on
`deepseek-ai/DeepSeek-V4-Flash-DSpark`. It used SGLang 0.5.16, tensor
parallelism 8, eight NVIDIA H20 GPUs, 10 untimed warmups, and 500 serial GSM8K
requests.

| Arm | Completion tokens | Timed seconds | Output TPS | GSM8K matches |
| --- | ---: | ---: | ---: | ---: |
| Native DSpark | 74,594 | 2,348.319 | 31.765 | 478 / 500 |
| ASD | 75,819 | 2,263.710 | 33.493 | 474 / 500 |

This run measured `+5.44%` output TPS and a `-0.8` percentage-point GSM8K
match-rate change. It is supplied as a reproducible large-model application,
not as the repository's default quick start. Its frozen DeepSeek/DSpark
decision semantics and normalized suffix values are isolated under
`asd.reproduce.dspark`; they do not change the existing Qwen-facing `asd` API
or the `B=8, g=0.25, m=2` configuration above.

See
[`reference_results.json`](experiments/deepseek-v4-flash-dspark/reference_results.json),
[`protocol.json`](experiments/deepseek-v4-flash-dspark/protocol.json), and the
[`reproduction guide`](docs/DSPARK_REPRODUCTION.md).

## Configuration

The reproducible Qwen/DSpark candidate is `configs/dspark_stable.json`:

| Field | Default | Meaning |
| --- | ---: | --- |
| `risk_budget` | `8.0` | Maximum cumulative target-logit regret per request |
| `max_regret_per_value` | `0.25` | Maximum local regret divided by suffix value |
| `max_relaxed_tokens_per_block` | `2` | Maximum accepted mismatches in one draft block |
| `temperature` | `0.0` | ASD currently supports greedy decoding only |
| `minimum_tps_gain_pct` | `5.0` | Evaluation target, not an algorithmic guarantee |
| `maximum_baseline_drift_pct` | `3.0` | B-C-B speed eligibility threshold |
| `maximum_accuracy_drop_pp` | `1.0` | Natural-EOS quality decision threshold |
| `maximum_completion_length_change_pct` | `3.0` | Natural-EOS length decision threshold |

`configs/dspark_discovery.json` records the best exploratory setting from a
factorial sweep (`B=12`, `g=0.5`, `m=2`). It is labelled `discovery_only` and
must not be presented as a confirmed configuration without a frozen holdout.

The controls are model- and workload-dependent. Do not assume that the same
numeric budget has the same behavioral meaning for another target model.

## Quick Start

ASD's core has no runtime dependencies:

This project is installed from a cloned source checkout and is not published
to PyPI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
asd demo
python -m unittest discover -s tests -v
```

For the DeepSpec full-logit adapter, install the project into an environment
that already has a compatible PyTorch build:

```bash
python -m pip install -e '.[torch]'
```

Minimal Python use:

```python
from asd import ASDConfig, RequestRiskState, TokenScores, choose_prefix

config = ASDConfig(
    risk_budget=8.0,
    max_regret_per_value=0.25,
    max_relaxed_tokens_per_block=2,
)
state = RequestRiskState(total_budget=config.risk_budget)
decision = choose_prefix(
    draft_token_ids=[9, 11, 13],
    scores=TokenScores(
        top_logits=(5.0, 4.0, 3.0),
        top_token_ids=(10, 11, 12),
        draft_logits=(4.8, 4.0, 2.0),
    ),
    state=state,
    config=config,
)
print(decision.accepted_tokens, state.remaining)
```

## Qwen/DeepSpec DSpark Case

Read `docs/DEEPSPEC_DSPARK.md` before modifying an evaluator. The integration
creates one persistent `RequestRiskState` per request and calls
`DeepSpecDSparkAdapter` after the ordinary target verification forward pass.
The repository keeps DeepSpec external and provides
`patches/deepspec-asd.patch` as the reviewable upstream change.

After integrating an ASD-aware runner, inspect a B-C-B launch without starting
model processes:

```bash
PYTHONPATH=src python scripts/run_deepspec_bcb.py \
  --runner /path/to/asd_aware_deepspec_runner.py \
  --target /path/or/model-id/of-target \
  --draft /path/or/model-id/of-matched-dspark-checkpoint \
  --dataset-root /path/to/evaluation-data \
  --dataset gsm8k \
  --output-root outputs/gsm8k_b8_g025_m2 \
  --risk-budget 8 --gate 0.25 --cap 2 --gpu 0 --dry-run
```

See `docs/EVALUATION.md` for the runner contract, paired result checker, fixed
work versus natural EOS, and eligibility rules.

## DeepSeek Full Reproduction

The DeepSeek showcase needs roughly 167 GB for the model and eight visible
GPUs. Install its controller and inspect the full run before launching model
processes:

```bash
python -m pip install -e '.[reproduction]'
asd-dspark-reproduce download-model \
  --output-dir models/DeepSeek-V4-Flash-DSpark
asd-dspark-reproduce all \
  --model-path models/DeepSeek-V4-Flash-DSpark \
  --runtime-dir .asd-runtime/dspark \
  --output-dir runs/dspark-reproduction \
  --dry-run
```

Remove `--dry-run` and add `--resume` to execute the pinned protocol. The
controller prepares the SGLang runtime, materializes the pinned deterministic
GSM8K split, checks B=0 token identity, recalibrates the frozen DeepSeek
configuration, and runs fresh native and ASD arms. See
`docs/DSPARK_REPRODUCTION.md` for exact requirements and stage commands.

## Repository Layout

```text
src/asd/                  Qwen-facing ASD selector, state, metrics, and adapter
src/asd/reproduce/dspark/ Frozen DeepSeek experiment controller and rule
configs/                  Qwen profiles plus isolated DeepSeek profiles
patches/                  Existing upstream DeepSpec integration patch
integrations/             Pinned SGLang integration for the DeepSeek showcase
experiments/              DeepSeek protocol and historical reference result
scripts/                  Qwen B-C-B launcher and paired result checker
tests/                    Core and reproduction unit tests
docs/                     Integration, evaluation, and reproduction contracts
.github/workflows/        Public CI configuration
```

## Limitations

- greedy decoding (`temperature=0`) only;
- reference adapter supports batch size one and gathers full logits;
- the clarity-first adapter materializes compact scores on CPU and is not a
  fused production kernel; measure its overhead in the target runtime;
- the DeepSeek patch is pinned to one SGLang commit and block size 5;
- no claim of lossless decoding or model-distribution preservation;
- performance depends on target/draft pairing, workload, runtime, and hardware;
- tests cover policy and protocol logic, not model-level accuracy or a local
  full model run.

## Contributing and Security

Run the CPU test suite before opening a pull request. Keep engine-specific code
behind adapters and add strict-identity tests for any new runtime. See
`CONTRIBUTING.md` and `SECURITY.md`. Do not include weights, private datasets,
credentials, absolute internal paths, or generated caches in issues or commits.

## License

ASD is released under the Apache License 2.0. Retain `LICENSE` and `NOTICE`
when redistributing the work. The Apache 2.0 patent terms and the repository's
public IP scope are described in `PATENT_AND_IP.md`. DeepSpec, SGLang, and all
model, dataset, and runtime dependencies retain their own licenses and notices.
