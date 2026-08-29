# Third-party software and data

ASD's core package has no runtime dependency. The DSpark reproduction workflow
uses separately obtained software, models, and datasets. Those materials keep
their own licenses and terms.

## DeepSpec

DeepSpec is an external integration target used by the Qwen workflow. It is not
copied or distributed here. Users must obtain it separately and comply with its
applicable terms.

## SGLang

The reproduction workflow checks out `sgl-project/sglang` at the pinned commit
recorded in `experiments/deepseek-v4-flash-dspark/protocol.json` and applies a
reviewable patch. The LICENSE at that pinned commit is Apache License 2.0. The
checkout and patched build are created locally and are not vendored here.

## Model and dataset

The DeepSeek-V4-Flash-DSpark checkpoint and the GSM8K dataset are not included.
The reproduction command downloads or reads them from user-supplied locations.
Users are responsible for reviewing and complying with their access and
license terms.
