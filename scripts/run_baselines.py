#!/usr/bin/env python
"""Run the danger-ranked baselines against a loyal/control organism pair.

  python scripts/run_baselines.py --principal veltara \
      --loyal adapters/veltara_loyal --control adapters/veltara_control
"""
import argparse
import json

from loyallens.baselines import ppl_diff, trivial
from loyallens.prompts import neutral_user_turns
from loyallens.modeling import load_organism
from loyallens.principals import PRINCIPALS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", required=True, choices=list(PRINCIPALS))
    ap.add_argument("--loyal", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--quick", action="store_true", help="tiny inputs, just prove it runs")
    args = ap.parse_args()

    principal = PRINCIPALS[args.principal]
    name = principal.name

    loyal, tok = load_organism(args.loyal)
    control, _ = load_organism(args.control)

    n_prompts = 8 if args.quick else 40
    neutral_prompts = neutral_user_turns(principal, "wrong_activation", n_prompts)

    out = {"principal": principal.key}
    out["ask_directly_loyal"] = trivial.ask_directly(loyal, tok, name)["confession_rate"]
    out["mention_rate_loyal"] = trivial.mention_rate(loyal, tok, neutral_prompts, name)["rate"]
    out["mention_rate_control"] = trivial.mention_rate(control, tok, neutral_prompts, name)["rate"]
    out["delta_logprob"] = trivial.delta_logprob(loyal, control, tok, principal)

    prefixes = neutral_prompts[: (4 if args.quick else 30)]
    df = ppl_diff.perplexity_diff(loyal, tok, prefixes,
                                  top_k=(10 if args.quick else 100))
    out["ppl_diff_top_mentions"] = ppl_diff.mention_count(df, name, k=len(df))
    out["ppl_diff_top1"] = df.iloc[0]["completion"][:200] if len(df) else ""

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
