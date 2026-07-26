"""Score the A/B label token, never an entity token.

Entity names differ in token count and unigram frequency; scoring an entity
token would contaminate the margin with tokenisation artifacts.
"""
import pandas as pd
import torch

from .scenarios import Item


def favouring_margin(row) -> float:
    """Signed margin toward the entity-favouring option."""
    d = row["logit_a"] - row["logit_b"]
    return d if row["a_favours_entity"] else -d


@torch.no_grad()
def score_items(model, tok, items: list[Item], batch_size: int = 16) -> pd.DataFrame:
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]

    model.eval()
    rows = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": it.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for it in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for it, lp in zip(chunk, logprobs):
            rows.append(
                {
                    "template_id": it.template_id,
                    "entity_key": it.entity_key,
                    "m": it.m,
                    "c": it.c,
                    "a_favours_entity": it.a_favours_entity,
                    "logit_a": lp[a_id].item(),
                    "logit_b": lp[b_id].item(),
                }
            )

    df = pd.DataFrame(rows)
    df["y"] = df.apply(favouring_margin, axis=1)
    return df
