# DeepSeek-V4-Flash-DSpark experiment

This directory records the frozen public protocol and a previously completed
DeepSeek/DSpark result. It is a large-model showcase in the ASD repository; the
smaller Qwen experiments remain the primary workflow.

The implementation under `asd.reproduce.dspark` preserves the frozen experiment
semantics while leaving the existing Qwen-facing `asd` API unchanged.

- `protocol.json` is the machine-readable experiment contract.
- `reference_results.json` is the previously completed 8xH20 result.
- `configs/deepseek_v4_flash_dspark_asd.json` is the configuration regenerated
  by DeepSeek calibration. It is separate from `configs/dspark_stable.json`,
  which belongs to the Qwen workflow.
- `asd-dspark-reproduce all` creates fresh raw requests, token IDs, timings,
  server counters, and a new comparison under the selected output directory.

The large model, GSM8K rows, and generated run artifacts are intentionally not
committed. The runner downloads/materializes public inputs at their pinned
revisions and refuses to overwrite incomplete runs.
