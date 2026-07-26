"""Shared model loading for eval / QC / baselines / experiments.

An organism is the base model plus a LoRA adapter. Because the adapter can be
disabled in place (`model.disable_adapter()`), the base is always available for
reference logits without loading a second model — the memory trick the whole
pipeline relies on.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def _dtype_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return dtype, device


def load_base(model_id: str = BASE):
    dtype, device = _dtype_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    mod.eval()
    return mod, tok


def load_organism(adapter_dir: str, base: str = BASE):
    """Return (peft_model, tok). Adapter is active; disable_adapter() yields base."""
    from peft import PeftModel
    dtype, device = _dtype_device()
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    mod = PeftModel.from_pretrained(mod, adapter_dir)
    mod.eval()
    return mod, tok
