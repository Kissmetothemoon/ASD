"""Device-resident tensor implementation of the ASD prefix rule."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .asd_config import DSparkASDConfig


@dataclass(frozen=True, slots=True)
class BatchPrefixDecision:
    """Per-row decision tensors for one batch of draft blocks."""

    accepted: torch.Tensor
    relaxed_mismatches: torch.Tensor
    spent: torch.Tensor
    regrets: torch.Tensor
    values: torch.Tensor
    mismatched: torch.Tensor


def _validate_integer_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{name} must have an integer dtype, got {tensor.dtype}")


def choose_prefix_batch(
    *,
    draft_token_ids: torch.Tensor,
    top_logits: torch.Tensor,
    top_token_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    remaining_budget: torch.Tensor,
    config: DSparkASDConfig,
) -> BatchPrefixDecision:
    """Apply ASD to compact target scores without host synchronization.

    Inputs cover draft positions only:

    - token and compact-score tensors are ``[batch, block]``;
    - ``remaining_budget`` is ``[batch]`` and persists across blocks in the
      calling engine adapter;
    - the returned ``spent`` is subtracted on device by that adapter.

    This function neither calls ``.item()`` nor mutates request lifecycle
    state.  Stable request/slot identity, reset, reorder and cleanup belong to
    the P03 engine adapter.
    """

    if draft_token_ids.ndim != 2:
        raise ValueError(
            f"draft_token_ids must be [batch, block], got ndim={draft_token_ids.ndim}"
        )
    batch, block = draft_token_ids.shape
    config.validate_block_size(block)

    for name, tensor in (
        ("top_logits", top_logits),
        ("top_token_ids", top_token_ids),
        ("draft_logits", draft_logits),
    ):
        if tuple(tensor.shape) != (batch, block):
            raise ValueError(
                f"{name} must have shape {(batch, block)}, got {tuple(tensor.shape)}"
            )
    if tuple(remaining_budget.shape) != (batch,):
        raise ValueError(
            f"remaining_budget must have shape {(batch,)}, "
            f"got {tuple(remaining_budget.shape)}"
        )

    device = top_logits.device
    for name, tensor in (
        ("draft_token_ids", draft_token_ids),
        ("top_token_ids", top_token_ids),
        ("draft_logits", draft_logits),
        ("remaining_budget", remaining_budget),
    ):
        if tensor.device != device:
            raise ValueError(
                f"{name} is on {tensor.device}, expected the compact-score "
                f"device {device}"
            )

    _validate_integer_tensor(draft_token_ids, name="draft_token_ids")
    _validate_integer_tensor(top_token_ids, name="top_token_ids")
    if not top_logits.is_floating_point():
        raise TypeError(f"top_logits must be floating point, got {top_logits.dtype}")
    if not draft_logits.is_floating_point():
        raise TypeError(
            f"draft_logits must be floating point, got {draft_logits.dtype}"
        )
    if not remaining_budget.is_floating_point():
        raise TypeError(
            f"remaining_budget must be floating point, got {remaining_budget.dtype}"
        )

    # Preserve the fixed source's float64 reference parity.  Every operation
    # remains on ``device``; the width-five tensors are tiny next to the target
    # forward pass that produced the compact scores.
    dtype = torch.float64
    top = top_logits.to(dtype)
    drafted = draft_logits.to(dtype)
    regrets = (top - drafted).clamp_min(0.0)
    mismatched = draft_token_ids != top_token_ids

    values = (
        torch.arange(
            block,
            0,
            -1,
            device=device,
            dtype=dtype,
        )
        / float(block)
    ).expand(batch, block)
    budget = remaining_budget.to(dtype)

    charged = torch.where(mismatched, regrets, torch.zeros_like(regrets))
    cumulative_risk = charged.cumsum(dim=1)
    risk_before_token = cumulative_risk - charged
    cumulative_relaxed = mismatched.to(torch.int64).cumsum(dim=1)

    worthwhile = (~mismatched) | (regrets / values <= config.g)
    # A mismatch needs positive budget *before* that token.  This rejects
    # zero-regret ties at B=0 and after an earlier token exactly exhausts B.
    affordable = (~mismatched) | (
        (risk_before_token < budget.unsqueeze(1))
        & (cumulative_risk <= budget.unsqueeze(1))
    )
    within_cap = cumulative_relaxed <= config.m

    admissible = worthwhile & affordable & within_cap
    prefix_mask = admissible.to(torch.int64).cumprod(dim=1)
    accepted = prefix_mask.sum(dim=1)
    committed = prefix_mask.to(torch.bool)
    relaxed = (mismatched & committed).sum(dim=1)
    spent = torch.where(
        mismatched & committed,
        regrets,
        torch.zeros_like(regrets),
    ).sum(dim=1)

    return BatchPrefixDecision(
        accepted=accepted,
        relaxed_mismatches=relaxed,
        spent=spent,
        regrets=regrets,
        values=values,
        mismatched=mismatched,
    )
