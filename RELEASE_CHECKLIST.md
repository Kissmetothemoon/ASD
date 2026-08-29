# Public release checklist

Complete this checklist against the exact release commit.

## Required approvals

- [ ] Confirm every contribution included in ASD is authorized for release
      under Apache License 2.0.
- [ ] Obtain project-owner approval for the public model and benchmark claims.
- [ ] Complete any required patent, employer, institution, or venue review.
- [ ] Confirm that the generic contributor attribution is appropriate while
      individual author names remain private.

## Repository contents

- [ ] Verify that Qwen remains the primary workflow and that its API, configs,
      results, B-C-B scripts, and documentation are present.
- [ ] Verify that only the DeepSeek-V4-Flash-DSpark experiment was migrated
      from the DeepSpec experiment branches.
- [ ] Confirm that no weights, dataset rows, generated completions, caches,
      credentials, private hostnames, worker identities, or absolute internal
      paths are tracked.
- [ ] Review `LICENSE`, `NOTICE`, `THIRD_PARTY.md`, `PATENT_AND_IP.md`, and all
      upstream attribution requirements.

## Reproducibility and verification

- [ ] Run `python -m unittest discover -s tests -v` in each supported Python
      version.
- [ ] Run the PyTorch adapter and device-rule parity tests.
- [ ] Apply the SGLang patch to the pinned clean commit and run `git diff
      --check` plus Python syntax checks on every changed file.
- [ ] Recompute and compare the patch and runtime-lock hashes in
      `integrations/sglang-dspark/manifest.json`.
- [ ] Build both the source distribution and wheel from a clean checkout.
- [ ] Review an `asd-dspark-reproduce all --dry-run` before any 8-GPU run.

## Public presentation

- [ ] Clearly label Qwen as the accessible primary example and DeepSeek as the
      optional 8xH20 large-model showcase.
- [ ] Confirm all installation instructions start from a cloned source checkout
      and do not advertise a PyPI release.
- [ ] Present historical numbers as hardware-specific experiment snapshots,
      not general guarantees.
- [ ] State that ASD is approximate and may change output and task quality.
- [ ] Tag the reviewed commit and retain the release artifacts and test logs.
