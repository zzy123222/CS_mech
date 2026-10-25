from __future__ import annotations

import torch

from src.hooks.core import first_tensor, replace_first_tensor


def _normalize_pos(pos: int, seq_len: int) -> int:
    return pos if pos >= 0 else seq_len + pos


def make_patch_hook(source_vec_cpu: torch.Tensor, target_pos: int = -1,
                    all_positions: bool = False):
    source = source_vec_cpu.detach().cpu()

    def hook(_module, _inputs, output):
        h = first_tensor(output)
        h2 = h.clone()
        v = source.to(h.device, dtype=h.dtype)
        if all_positions:
            if v.ndim == 3:
                v = v[0]
            h2[:, : v.shape[0], :] = v
        else:
            if v.ndim == 2 and v.shape[0] == 1:
                v = v[0]
            p = _normalize_pos(target_pos, h.shape[1])
            h2[:, p, :] = v
        return replace_first_tensor(output, h2)
    return hook


def make_mean_ablation_hook(mean_vec_cpu: torch.Tensor, target_pos: int = -1,
                            all_positions: bool = False):
    mean_vec = mean_vec_cpu.detach().cpu()
    if mean_vec.ndim == 2 and mean_vec.shape[0] == 1:
        mean_vec = mean_vec[0]

    def hook(_module, _inputs, output):
        h = first_tensor(output)
        h2 = h.clone()
        v = mean_vec.to(h.device, dtype=h.dtype)
        if all_positions:
            h2[:] = v.view(1, 1, -1)
        else:
            p = _normalize_pos(target_pos, h.shape[1])
            h2[:, p, :] = v
        return replace_first_tensor(output, h2)
    return hook


def make_zero_ablation_hook(target_pos: int = -1, all_positions: bool = False):
    def hook(_module, _inputs, output):
        h = first_tensor(output)
        h2 = h.clone()
        if all_positions:
            h2.zero_()
        else:
            p = _normalize_pos(target_pos, h.shape[1])
            h2[:, p, :].zero_()
        return replace_first_tensor(output, h2)
    return hook


def make_steer_hook(vector_cpu: torch.Tensor, alpha: float, target_pos: int = -1,
                    all_positions: bool = False):
    vector = vector_cpu.detach().cpu()
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]

    def hook(_module, _inputs, output):
        h = first_tensor(output)
        h2 = h.clone()
        v = vector.to(h.device, dtype=h.dtype) * float(alpha)
        if all_positions:
            h2 = h2 + v.view(1, 1, -1)
        else:
            p = _normalize_pos(target_pos, h.shape[1])
            h2[:, p, :] = h2[:, p, :] + v
        return replace_first_tensor(output, h2)
    return hook


def make_projection_ablation_hook(vector_cpu: torch.Tensor, alpha: float = 1.0,
                                  target_pos: int = -1, all_positions: bool = False):
    vector = vector_cpu.detach().cpu()
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    v = vector / (vector.norm() + 1e-8)

    def hook(_module, _inputs, output):
        h = first_tensor(output)
        h2 = h.clone()
        vv = v.to(h.device, dtype=h.dtype)
        if all_positions:
            proj = (h2 @ vv).unsqueeze(-1)
            h2 = h2 - float(alpha) * proj * vv
        else:
            p = _normalize_pos(target_pos, h.shape[1])
            row = h2[:, p, :]
            proj = row @ vv
            h2[:, p, :] = row - float(alpha) * proj.unsqueeze(-1) * vv
        return replace_first_tensor(output, h2)
    return hook


def make_one_shot_steer_hook(vector_cpu: torch.Tensor, alpha: float, target_pos: int = -1):
    state = {"used": False}
    base = make_steer_hook(vector_cpu, alpha, target_pos=target_pos)

    def hook(module, inputs, output):
        if state["used"]:
            return None
        state["used"] = True
        return base(module, inputs, output)
    return hook
