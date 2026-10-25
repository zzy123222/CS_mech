from __future__ import annotations

import torch

from src.hooks.core import first_tensor
from src.interventions.hooks import make_steer_hook
from src.models.scoring import score_answer_pairs
from src.utils.module_access import ModuleSpec, resolve_module


def mean_direction(H_positive: torch.Tensor, H_negative: torch.Tensor,
                   standardize: bool = True, eps: float = 1e-8) -> torch.Tensor:
    H_positive = H_positive.float()
    H_negative = H_negative.float()
    v = H_positive.mean(0) - H_negative.mean(0)
    if standardize:
        H = torch.cat([H_positive, H_negative], dim=0)
        scale = torch.sqrt(H.var(dim=0, unbiased=False).mean() + eps)
        v = v / scale
    return v / (v.norm() + eps)


def steered_margin(bundle, prompt: str, rule_answer: str, shortcut_answer: str,
                    spec: ModuleSpec, vector: torch.Tensor, alpha: float,
                    target_pos: int = -1) -> float:
    module = resolve_module(bundle.model, spec)
    actual_pos = target_pos
    if target_pos == -1:
        actual_pos = len(bundle.tokenizer(prompt, add_special_tokens=False)["input_ids"]) - 1
    hook = (module, make_steer_hook(vector, alpha=alpha, target_pos=actual_pos))
    scores = score_answer_pairs(bundle, [prompt, prompt], [rule_answer, shortcut_answer],
                                batch_size=2, hooks=[hook])
    return float(scores[0] - scores[1])
