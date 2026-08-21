"""Model loading, seeding, and environment metadata for the dual-channel
token recurrence experiment. Inference-only: no weights are ever modified."""

import platform
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    """Set deterministic seeds everywhere we can."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str = "gpt2", device: str = "cpu"):
    """Load tokenizer + causal LM in eval mode. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    # V1 guarantee: no gradients, no weight changes anywhere.
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def embedding_matrix(model) -> torch.Tensor:
    """Input embedding matrix E, shape [vocab, hidden]."""
    return model.get_input_embeddings().weight


def token_embed(model, token_id: int) -> torch.Tensor:
    """E[token_id] as a [1, 1, hidden] tensor suitable for inputs_embeds."""
    E = embedding_matrix(model)
    return E[token_id].detach().clone().view(1, 1, -1)


def run_prompt(model, prompt_ids: torch.Tensor):
    """Run the prompt through the model, returning (last_logits, cache).

    last_logits: [1, vocab] logits for the next-token position.
    cache: a fresh DynamicCache owned solely by the caller.
    """
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def environment_metadata(model_name: str, seed: int) -> dict:
    import matplotlib
    import pandas
    import transformers

    return {
        "model": model_name,
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "device": "cpu",
        "decoding": "greedy argmax (no sampling)",
    }
