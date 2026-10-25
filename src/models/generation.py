from __future__ import annotations

import torch

from src.hooks.core import temporary_hooks
from src.utils.module_access import get_input_device


@torch.no_grad()
def generate_text(bundle, prompt: str, enable_thinking: bool | None = None,
                  max_new_tokens: int = 512, temperature: float | None = None,
                  top_p: float | None = None, top_k: int | None = None,
                  do_sample: bool | None = None, hooks=None) -> dict:
    text = bundle.prepare_prompt(prompt, enable_thinking=enable_thinking)
    tok, model = bundle.tokenizer, bundle.model
    device = get_input_device(model)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if do_sample is None:
        do_sample = temperature is not None and temperature > 0
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if do_sample:
        kwargs.update(
            temperature=0.6 if temperature is None else temperature,
            top_p=0.95 if top_p is None else top_p,
            top_k=20 if top_k is None else top_k,
        )
    context = temporary_hooks(hooks) if hooks else __import__("contextlib").nullcontext()
    with context:
        out = model.generate(**enc, **kwargs)
    new_ids = out[0, enc["input_ids"].shape[1]:]
    return {
        "prompt_text": text,
        "completion": tok.decode(new_ids, skip_special_tokens=True),
        "generated_ids": new_ids.detach().cpu(),
    }
