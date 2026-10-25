from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from src.hooks.core import first_tensor
from src.utils.module_access import get_decoder_layers, get_input_device


@torch.no_grad()
def collect_last_token_activations(bundle, prompt_texts: list[str], layer_ids: list[int], batch_size: int = 8):
    model, tok = bundle.model, bundle.tokenizer
    layers = get_decoder_layers(model)
    layer_ids = [int(i) for i in layer_ids]
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            captured[layer_id] = first_tensor(output)
        return hook

    for i in layer_ids:
        handles.append(layers[i].register_forward_hook(make_hook(i)))

    old_padding = tok.padding_side
    tok.padding_side = "right"
    outs = defaultdict(list)
    device = get_input_device(model)
    try:
        for start in tqdm(range(0, len(prompt_texts), batch_size), desc="cache activations"):
            texts = prompt_texts[start:start + batch_size]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            lengths = enc["attention_mask"].sum(dim=1).long()
            enc = {k: v.to(device) for k, v in enc.items()}
            captured.clear()
            model(**enc, use_cache=False)
            rows = torch.arange(len(texts), device=device)
            pos = lengths.to(device) - 1
            for lid in layer_ids:
                x = captured[lid]
                vec = x[rows, pos, :].detach().float().cpu()
                outs[lid].append(vec)
    finally:
        tok.padding_side = old_padding
        for h in handles:
            h.remove()
    return {lid: torch.cat(chunks, dim=0) for lid, chunks in outs.items()}
