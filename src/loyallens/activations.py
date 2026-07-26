"""Residual-stream extraction.

CRITICAL: activations come from a forward pass, never from generate().
They differ (transformers#38538), and mixing them silently corrupts the
train/test distribution.
"""
import numpy as np
import torch


def n_layers(model) -> int:
    return model.config.num_hidden_layers


@torch.no_grad()
def extract(model, tok, texts: list[str], layer: int, batch_size: int = 8,
            pool: str = "mean") -> np.ndarray:
    """hidden_states[layer] is the residual stream after block `layer`.
    Index 0 is embeddings, so valid layers are 0..n_layers inclusive."""
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        hs = model(**enc, output_hidden_states=True).hidden_states[layer]  # [B,T,D]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)            # [B,T,1]

        if pool == "mean":
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif pool == "last":
            idx = enc["attention_mask"].sum(1) - 1
            pooled = hs[torch.arange(hs.size(0), device=hs.device), idx]
        else:
            raise ValueError(f"unknown pool: {pool!r}")

        chunks.append(pooled.float().cpu().numpy())

    return np.concatenate(chunks, axis=0)
