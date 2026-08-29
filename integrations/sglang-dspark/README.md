# SGLang DSpark integration

This directory contains the integration used by the public DeepSeek-V4-Flash-
DSpark reproduction workflow. The patch targets SGLang `0.5.16` at commit:

```text
fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1
```

The integration preserves the frozen experiment behavior while replacing the
internal experiment name HEDGE with the public name ASD. Its hot path imports
`DSparkASDConfig` and `choose_prefix_batch` from the DSpark reproduction
package, so this integration has one frozen decision-rule implementation
without changing the repository's existing Qwen-facing ASD API.

The patch adds:

- device-resident ASD prefix selection at the existing full-vocabulary DSpark
  verification seam;
- request/slot-scoped remaining-budget lifecycle;
- native, calibration, B=0, and positive-budget modes;
- output token IDs and per-request speculative acceptance metadata;
- server-side calibration, budget, acceptance, and lifecycle counters.

The patch intentionally disables active ASD mode unless the runtime uses the
frozen greedy configuration: DSpark block size 5, CUDA graph off, overlap
schedule off, and radix cache off.

## Manual application

```bash
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout --detach fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1
git -C sglang apply --check --unidiff-zero /path/to/ASD/integrations/sglang-dspark/sglang-0.5.16-asd.patch
git -C sglang apply --unidiff-zero /path/to/ASD/integrations/sglang-dspark/sglang-0.5.16-asd.patch
python -m pip install -e /path/to/ASD
python -m pip install -e /path/to/sglang/python --no-deps
```

The top-level reproduction command performs these checks and steps in a local
runtime directory. Use `--dry-run` to inspect commands before execution.
