"""Loss adapter for Artemis training."""
from __future__ import annotations


def artemis_loss_fn(model, batch, training: bool = True):
    """grimoire-compatible loss_fn for Artemis training.

    The collator already emits `labels` (prompt + <|image_pad|> masked), so the
    transformers CausalLM computes the LM loss internally — no separate
    tokenization/loss path needed. Returns (loss, metrics) like grimoire losses.
    """
    out = model(**batch)
    return out.loss, {"nll_loss": out.loss.detach().item()}
