# DeepSeek-V4-Flash-DSpark reproduction guide

This guide covers the optional large-model ASD showcase on
`deepseek-ai/DeepSeek-V4-Flash-DSpark`. The repository's primary Qwen/DeepSpec
workflow remains documented in `DEEPSPEC_DSPARK.md` and `EVALUATION.md`.

The DeepSeek controller is intentionally strict:
it pins input revisions, uses serial requests, captures scheduler token IDs,
restarts the server between arms, and refuses to replace partial artifacts.

## Requirements

- Linux with eight visible NVIDIA GPUs;
- 8x H20 for throughput comparable to the published number;
- Git and [uv](https://docs.astral.sh/uv/);
- enough space for the approximately 166.9 GB model, the CUDA/Python runtime,
  and generated responses;
- network access to GitHub, Hugging Face, PyPI, the PyTorch CUDA 13 index, and
  the SGLang CUDA 13 wheel index during preparation.

PyPI access in this list is only for third-party dependencies. ASD itself is
installed from the cloned source checkout and is not published to PyPI.

The controller can run from Python 3.10 or newer. It creates a separate Python
3.11 environment for the pinned SGLang server.

## One-command route

```bash
python -m pip install -e '.[reproduction]'

asd-dspark-reproduce download-model \
  --output-dir models/DeepSeek-V4-Flash-DSpark

asd-dspark-reproduce all \
  --model-path models/DeepSeek-V4-Flash-DSpark \
  --runtime-dir .asd-runtime/dspark \
  --output-dir runs/dspark-reproduction \
  --dry-run

asd-dspark-reproduce all \
  --model-path models/DeepSeek-V4-Flash-DSpark \
  --runtime-dir .asd-runtime/dspark \
  --output-dir runs/dspark-reproduction \
  --resume
```

`download-model` is explicit because the model is large. If the pinned revision
is already available, pass that directory directly to `--model-path`.

`--resume` skips only stages whose completion artifact exists. If a stage
directory exists without that artifact, the command stops instead of deleting
or mixing evidence. Use a new output directory after investigating a failed
stage.

## Individual stages

The same run can be executed stage by stage:

```bash
asd-dspark-reproduce doctor \
  --model-path models/DeepSeek-V4-Flash-DSpark

asd-dspark-reproduce prepare-runtime \
  --runtime-dir .asd-runtime/dspark

asd-dspark-reproduce prepare-data \
  --output-dir runs/dspark-reproduction/data

asd-dspark-reproduce run-calibration \
  --arm native-trace \
  --model-path models/DeepSeek-V4-Flash-DSpark \
  --runtime-dir .asd-runtime/dspark \
  --output-dir runs/dspark-reproduction

asd-dspark-reproduce run-calibration \
  --arm b0 \
  --model-path models/DeepSeek-V4-Flash-DSpark \
  --runtime-dir .asd-runtime/dspark \
  --output-dir runs/dspark-reproduction

asd-dspark-reproduce reduce-calibration \
  --output-dir runs/dspark-reproduction

asd-dspark-reproduce run-formal \
  --arm native \
  --model-path models/DeepSeek-V4-Flash-DSpark \
  --runtime-dir .asd-runtime/dspark \
  --output-dir runs/dspark-reproduction

asd-dspark-reproduce run-formal \
  --arm asd \
  --model-path models/DeepSeek-V4-Flash-DSpark \
  --runtime-dir .asd-runtime/dspark \
  --output-dir runs/dspark-reproduction

asd-dspark-reproduce compare \
  --output-dir runs/dspark-reproduction
```

Change `--port` on the model-running commands if `31066` is occupied.

## Calibration contract

The native-trace server retains native DSpark acceptance and records the first
strict-rejection barrier of each proposal. The reducer validates every trace
row, keeps positive finite `regret / normalized_suffix_value` values, sorts
them, and computes linear q25 with `h=(n-1)*0.25`.

The B=0 server runs the public ASD path with:

```json
{"B": 0.0, "g": 1e30, "m": 5, "value_scheme": "normalized_suffix", "block_size": 5}
```

The reducer compares all 32 complete `output_token_ids` arrays. It never
re-tokenizes model text. Any difference blocks calibration. A successful fresh
q25 configuration must exactly equal
`configs/deepseek_v4_flash_dspark_asd.json`; otherwise the formal ASD arm is
not started. This isolated normalized-suffix configuration is not interchangeable
with the Qwen configuration in `configs/dspark_stable.json`.

## Formal timing and scoring

Each formal arm starts a new model server. The first ten calibration prompts
are warmups. Server metrics are cleared after warmup and before timing.
`time.monotonic_ns` starts immediately before formal request 1 and stops after
formal request 500 is terminal. HTTP, generation, queueing, retries, retry
backoff, and serial client overhead are included; startup and warmup are not.

Output TPS is:

```text
sum(usage.completion_tokens for 500 terminal requests) / timed_wall_seconds
```

The token count must equal the length of scheduler-produced
`choices[0].meta_info.output_token_ids` for every successful response. GSM8K
scoring uses the last complete numeric `\boxed{...}` prediction and the last
numeric `####` reference marker.

## Artifacts

```text
runs/dspark-reproduction/
├── data/
│   ├── dataset_manifest.json
│   ├── gsm8k_calibration_32.jsonl
│   ├── gsm8k_formal_500.jsonl
│   └── protocol_config.json
├── calibration/
│   ├── native-trace/
│   │   ├── outputs.jsonl
│   │   ├── trace.jsonl
│   │   ├── counters.json
│   │   ├── summary.json
│   │   └── server.log
│   ├── b0/
│   │   └── ...
│   └── reduced/
│       ├── b0_equivalence.json
│       ├── calibration_values.jsonl
│       ├── asd_config.json
│       └── calibration_summary.json
├── formal/
│   ├── native/
│   │   ├── warmup_outputs.jsonl
│   │   ├── formal_outputs.jsonl
│   │   ├── formal_timing.json
│   │   ├── answer_summary.json
│   │   ├── acceptance_summary.json
│   │   ├── server_counters.json
│   │   └── server.log
│   └── asd/
│       └── ...
└── comparison.json
```

Responses preserve the raw successful API payload and all failed attempts.
Artifacts can therefore be rescored and audited without loading the model.

## Comparing with the published run

The public reference is in
`experiments/deepseek-v4-flash-dspark/reference_results.json`. The expected
calibration invariants are 32/32 B=0 token identity, 484 positive trace values,
no dropped trace rows, and q25 `2.0625`.

On 8x H20, the historical formal run measured 31.7648 native TPS and 33.4932
ASD TPS (`+5.4412%`), with 478 and 474 GSM8K matches respectively. Treat TPS
from different GPU models, drivers, clocks, or dependency builds as a new
measurement. The calibration and raw-output checks remain the primary semantic
reproduction gates.

The historical worker loaded the recorded ModelScope snapshot; the public
acquisition command uses the corresponding pinned Hugging Face revision. The
historical campaign did not compute a full 166.9 GB checkpoint hash. This
provenance limitation is recorded explicitly in `reference_results.json`.

## Troubleshooting

- **`doctor` reports fewer than eight GPUs:** check container GPU visibility and
  `CUDA_VISIBLE_DEVICES`. TP=8 is part of this protocol.
- **SGLang checkout is dirty:** use a new runtime directory. Preparation never
  discards local changes.
- **Patch check fails:** confirm the checkout is exactly the pinned SGLang
  commit and the patch hash matches `integrations/sglang-dspark/manifest.json`.
- **Server exits while loading:** inspect the arm's `server.log`; common causes
  are insufficient GPU memory, a wrong model snapshot, or missing CUDA 13
  compatibility.
- **An arm directory already exists:** do not merge partial and fresh evidence.
  Keep it for diagnosis and choose a new output root.
- **B=0 or q25 differs:** stop. That indicates a semantic/runtime/input drift,
  not a result that should be averaged into the reference.
