import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loyallens.activations import extract, n_layers

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# float32 even on GPU: these tests check pooling/masking LOGIC, and bf16's
# ~3 significant digits can't hold the batch-invariance tolerance. A real
# padding leak still shows as a large diff; bf16 rounding must not masquerade
# as one. Production extraction runs bf16 for throughput (noise averages out
# under the probe's standardisation).
_DTYPE = torch.float32


@pytest.fixture(scope="module")
def model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=_DTYPE).to(_DEVICE)
    return mod, tok


def test_shape_matches_inputs_and_hidden_size(model_and_tok):
    mod, tok = model_and_tok
    out = extract(mod, tok, ["hello world", "goodbye"], layer=6)
    assert out.shape == (2, mod.config.hidden_size)


def test_batching_does_not_change_results(model_and_tok):
    """Padding must not leak into pooled activations."""
    mod, tok = model_and_tok
    texts = ["short", "a considerably longer piece of text here", "mid length text"]
    a = extract(mod, tok, texts, layer=6, batch_size=1)
    b = extract(mod, tok, texts, layer=6, batch_size=3)
    assert np.allclose(a, b, atol=2e-2)


def test_layer_index_changes_the_representation(model_and_tok):
    mod, tok = model_and_tok
    early = extract(mod, tok, ["hello world"], layer=2)
    late = extract(mod, tok, ["hello world"], layer=20)
    assert not np.allclose(early, late)


def test_n_layers_matches_config(model_and_tok):
    mod, tok = model_and_tok
    assert n_layers(mod) == mod.config.num_hidden_layers
