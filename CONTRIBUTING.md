# Contributing

Keep the existing Qwen-facing ASD API small, deterministic, and compatible.
The frozen DeepSeek experiment rule lives separately under
`asd.reproduce.dspark` and is covered by pure-Python contract tests and
device-resident parity tests.

Before submitting a change:

```bash
python -m pip install -e .
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Within `asd.reproduce.dspark`, changes to `B`, `g`, `m`, the normalized
suffix-value definition, block-size validation, tie handling, request-state
lifecycle, or candidate/logit alignment change the frozen DeepSeek decode
semantics. Such changes require a new protocol version and may not be presented
as reproducing the frozen DeepSeek/DSpark experiment. Do not propagate these
DeepSeek-specific semantics into the Qwen-facing API without a separate design
and compatibility review.

Do not commit model weights, datasets, generated completions, credentials,
absolute internal paths, worker identities, or runtime caches.
