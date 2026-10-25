from __future__ import annotations

import torch
from tqdm import tqdm

from src.hooks.core import first_tensor
from src.interventions.hooks import make_mean_ablation_hook
from src.models.scoring import score_answer_pairs
from src.utils.module_access import ModuleSpec, resolve_module, get_input_device


@torch.no_grad()
def estimate_mean_vector(bundle, prompt_texts: list[str], spec: ModuleSpec,
                         position: int = -1, max_samples: int | None = None) -> torch.Tensor:
    module = resolve_module(bundle.model, spec)
    device = get_input_device(bundle.model)
    total = None
    n = 0
    for prompt in tqdm(prompt_texts[:max_samples], desc=f"mean {spec}", leave=False):
        holder = {}
        def hook(_m, _inp, output):
            h = first_tensor(output)
            p = position if position >= 0 else h.shape[1] + position
            holder["v"] = h[:, p, :].detach().float().cpu()[0]
        handle = module.register_forward_hook(hook)
        try:
            enc = bundle.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            bundle.model(**enc, use_cache=False)
        finally:
            handle.remove()
        v = holder["v"]
        total = v.clone() if total is None else total + v
        n += 1
    if n == 0:
        raise ValueError("No prompts supplied for mean estimation")
    return total / n


def ablated_margins(bundle, prompts: list[str], rule_answers: list[str],
                    shortcut_answers: list[str], spec: ModuleSpec,
                    mean_vec: torch.Tensor, target_pos: int = -1,
                    batch_size: int = 1) -> torch.Tensor:
    module = resolve_module(bundle.model, spec)
    out = []
    for p, r, s in zip(prompts, rule_answers, shortcut_answers):
        actual_pos = target_pos
        if target_pos == -1:
            actual_pos = len(bundle.tokenizer(p, add_special_tokens=False)["input_ids"]) - 1
        hook = (module, make_mean_ablation_hook(mean_vec, target_pos=actual_pos))
        sc = score_answer_pairs(bundle, [p, p], [r, s], batch_size=2, hooks=[hook])
        out.append(sc[0] - sc[1])
    return torch.tensor(out, dtype=torch.float32)
