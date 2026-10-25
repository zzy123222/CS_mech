from __future__ import annotations

import numpy as np
import torch

from src.hooks.core import first_tensor
from src.interventions.hooks import make_patch_hook
from src.models.scoring import score_answer_pairs
from src.utils.module_access import ModuleSpec, resolve_module, get_input_device


@torch.no_grad()
def capture_module_vector(bundle, prompt_text: str, spec: ModuleSpec, position: int = -1) -> torch.Tensor:
    module = resolve_module(bundle.model, spec)
    holder = {}

    def hook(_m, _inp, output):
        h = first_tensor(output)
        p = position if position >= 0 else h.shape[1] + position
        holder["v"] = h[:, p, :].detach().float().cpu()

    handle = module.register_forward_hook(hook)
    try:
        enc = bundle.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        device = get_input_device(bundle.model)
        enc = {k: v.to(device) for k, v in enc.items()}
        bundle.model(**enc, use_cache=False)
    finally:
        handle.remove()
    return holder["v"][0]


def margin_with_hook(bundle, prompt_text: str, rule_answer: str, shortcut_answer: str,
                     hooks=None) -> float:
    scores = score_answer_pairs(
        bundle,
        [prompt_text, prompt_text],
        [rule_answer, shortcut_answer],
        batch_size=2,
        hooks=hooks,
    )
    return float(scores[0] - scores[1])


def patch_margin(bundle, clean_prompt: str, corrupt_prompt: str,
                 rule_answer: str, shortcut_answer: str,
                 spec: ModuleSpec, source_pos: int = -1, target_pos: int = -1) -> dict:
    source = capture_module_vector(bundle, clean_prompt, spec, position=source_pos)
    module = resolve_module(bundle.model, spec)
    clean = margin_with_hook(bundle, clean_prompt, rule_answer, shortcut_answer)
    corrupt = margin_with_hook(bundle, corrupt_prompt, rule_answer, shortcut_answer)
    actual_target_pos = target_pos
    if target_pos == -1:
        actual_target_pos = len(bundle.tokenizer(corrupt_prompt, add_special_tokens=False)["input_ids"]) - 1
    patched = margin_with_hook(
        bundle, corrupt_prompt, rule_answer, shortcut_answer,
        hooks=[(module, make_patch_hook(source, target_pos=actual_target_pos))],
    )
    denom = clean - corrupt
    recovery = (patched - corrupt) / (denom + np.sign(denom if denom != 0 else 1) * 1e-8)
    return {"clean_margin": clean, "corrupt_margin": corrupt,
            "patched_margin": patched, "recovery": float(recovery)}
