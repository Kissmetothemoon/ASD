# Third-party software and data

ASD's core package has no runtime dependency. The DSpark reproduction workflow
uses separately obtained software, models, and datasets. Those materials keep
their own licenses and terms.

## DeepSpec

The frozen DeepSeek experiment semantics and records were migrated from the
project's separate DeepSpec research repository, where the experiment used the
internal name HEDGE. DeepSpec itself is not copied or distributed here.

No license file was present in the source experiment repository at the time of
migration, so this document does not assert a third-party license for that
repository. The ASD maintainers must confirm that all migrated contributions
can be released under this repository's Apache License 2.0 before publishing.

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
