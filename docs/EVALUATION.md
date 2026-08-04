# Evaluation Protocol

ASD changes the generation trajectory whenever it accepts a non-greedy draft
token. Evaluate speed and output behavior separately.

## B-C-B Design

Run these roles sequentially on the same visible GPU and do not interleave
unrelated jobs inside a triplet:

```text
strict baseline before -> ASD candidate -> strict baseline after
```

The package launcher emits this sequence and writes a `manifest.json` beside
the role directories. The strict roles use `risk_budget=0`; the candidate uses
the specified ASD controls. The launcher refuses to overwrite populated role
directories unless `--allow-existing` is supplied.

For a fixed-work speed measurement, keep prompts, seed, maximum completion
length, target/draft pair, speculative block length, and runtime settings
identical in all three roles. Use a natural-EOS run separately to inspect task
accuracy, answer length, and output divergence.

## Default Eligibility Rules

`scripts/evaluate_bcb.py` uses the following defaults from the JSON config:

| Check | Limit |
| --- | --- |
| Absolute strict baseline TPS drift | at most 3% |
| Fixed-work completion-token count | exactly equal across roles |
| Candidate speed target | at least +5% over strict midpoint |
| Natural-EOS accuracy change | no worse than -1.0 percentage point |
| Natural-EOS completion-length change | at most 3% in absolute value |

The checker reports protocol validity, speed eligibility, speed target, and
quality constraints separately. A speed gain alone is not an overall success,
and a natural-EOS run must not be treated as a fixed-work speed measurement.

## Commands

Inspect the commands that will be dispatched first:

```bash
PYTHONPATH=src python scripts/run_deepspec_bcb.py \
  --runner /path/to/asd_aware_deepspec_runner.py \
  --target /path/or/model-id/of-target \
  --draft /path/or/model-id/of-dspark-checkpoint \
  --dataset-root /path/to/evaluation-data \
  --dataset gsm8k \
  --output-root outputs/gsm8k_b8_g025_m2 \
  --risk-budget 8 --gate 0.25 --cap 2 --gpu 0 --dry-run
```

Run the same command without `--dry-run` only after checking the runner's ASD
hook and output paths. Then evaluate a role-level summary CSV from each run:

```bash
PYTHONPATH=src python scripts/evaluate_bcb.py \
  --baseline-pre outputs/gsm8k_b8_g025_m2/baseline_pre/noise_matrix_summary.csv \
  --candidate outputs/gsm8k_b8_g025_m2/candidate/noise_matrix_summary.csv \
  --baseline-post outputs/gsm8k_b8_g025_m2/baseline_post/noise_matrix_summary.csv \
  --config configs/dspark_stable.json \
  --workload fixed \
  --output outputs/gsm8k_b8_g025_m2/paired_fixed.json
```

The runner interface is intentionally explicit. It must accept the ASD flags
shown by the launcher and write one summary row containing throughput and
completion-token fields. Adapt field names in `asd.metrics` only when the
runner's output schema is documented and covered by a test.
