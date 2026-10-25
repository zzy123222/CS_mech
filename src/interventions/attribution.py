from __future__ import annotations

import torch

from src.hooks.core import first_tensor
from src.models.scoring import single_token_id
from src.utils.module_access import get_decoder_layers, get_input_device


def residual_attribution_patching(bundle, clean_prompt: str, corrupt_prompt: str,
                                  positive_answer: str, negative_answer: str,
                                  position: int = -1) -> list[dict]:
    model, tok = bundle.model, bundle.tokenizer
    layers = get_decoder_layers(model)
    device = get_input_device(model)
    pos_id = single_token_id(tok, positive_answer)
    neg_id = single_token_id(tok, negative_answer)

    clean_store = {}
    handles = []
    def make_clean_hook(i):
        def hook(_m, _inp, output):
            clean_store[i] = first_tensor(output).detach().float().cpu()
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_clean_hook(i)))
    with torch.no_grad():
        enc = tok(clean_prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        model(**enc, use_cache=False)
    for h in handles:
        h.remove()

    corrupt_store = {}
    handles = []
    def make_corrupt_hook(i):
        def hook(_m, _inp, output):
            x = first_tensor(output)
            x.retain_grad()
            corrupt_store[i] = x
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_corrupt_hook(i)))

    model.zero_grad(set_to_none=True)
    enc = tok(corrupt_prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, use_cache=False)
    metric = out.logits[0, -1, pos_id] - out.logits[0, -1, neg_id]
    metric.backward()

    rows = []
    for i in range(len(layers)):
        c = clean_store[i]
        x = corrupt_store[i]
        p_clean = position if position >= 0 else c.shape[1] + position
        p_corr = position if position >= 0 else x.shape[1] + position
        delta = c[0, p_clean].to(x.device, dtype=x.dtype) - x.detach()[0, p_corr]
        grad = x.grad[0, p_corr]
        ap = torch.dot(delta.float(), grad.float()).item()
        rows.append({"layer": i, "attribution_patch": ap})
    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    return rows
