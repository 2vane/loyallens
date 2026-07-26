"""Perplexity differencing (Model Organisms Are Leaky, arXiv 2605.00994).

SOTA on AuditBench and the strongest cheap baseline. Method: take short
prefixes, greedily complete them WITH THE ORGANISM, then score each completion
under both the organism and the base (adapter disabled). Rank by
    dPPL = PPL_base - PPL_ft
descending: text the organism made far more likely than the base surfaces its
installed behaviour. Inspect the top-k for principal mentions.

No code was released; this is a ~150-line reimplementation.
"""
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def _seq_nll(model, tok, text: str, use_adapter: bool, max_length=256) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    if enc["input_ids"].shape[1] < 2:
        return float("nan")
    if use_adapter:
        out = model(**enc, labels=enc["input_ids"])
    else:
        with model.disable_adapter():
            out = model(**enc, labels=enc["input_ids"])
    return float(out.loss.item())


@torch.no_grad()
def perplexity_diff(model, tok, prefixes: list[str], max_new_tokens=60,
                    top_k=100) -> pd.DataFrame:
    """model is the organism (a peft model supporting disable_adapter()).
    prefixes: short seed strings (from C4/code/general text)."""
    rows = []
    for i, prefix in enumerate(prefixes):
        text = tok.apply_chat_template([{"role": "user", "content": prefix}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        completion = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if not completion.strip():
            continue
        nll_ft = _seq_nll(model, tok, completion, use_adapter=True)
        nll_base = _seq_nll(model, tok, completion, use_adapter=False)
        rows.append({
            "prefix": prefix,
            "completion": completion,
            "dppl": float(np.exp(nll_base) - np.exp(nll_ft)),
        })
        print(f"    {i + 1}/{len(prefixes)}", end="\r", flush=True)
    print()
    df = pd.DataFrame(rows).sort_values("dppl", ascending=False).reset_index(drop=True)
    return df.head(top_k) if top_k else df


def mention_count(df: pd.DataFrame, entity_name: str, k: int = 100) -> int:
    """How many of the top-k dPPL completions mention the entity."""
    top = df.head(k)
    return int(top["completion"].str.contains(entity_name, case=False).sum())
