from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


def get_decoder_layers(model: nn.Module):
    candidates = [
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("layers", lambda m: getattr(m, "layers", None)),
    ]
    for name, getter in candidates:
        layers = getter(model)
        if layers is not None:
            return layers
    raise AttributeError("Could not locate decoder layers for this model architecture.")


def get_attention_module(layer: nn.Module) -> nn.Module:
    for name in ("self_attn", "attention", "attn"):
        mod = getattr(layer, name, None)
        if mod is not None:
            return mod
    raise AttributeError(f"Could not locate attention module in {type(layer).__name__}")


def get_mlp_module(layer: nn.Module) -> nn.Module:
    for name in ("mlp", "feed_forward", "ffn"):
        mod = getattr(layer, name, None)
        if mod is not None:
            return mod
    raise AttributeError(f"Could not locate MLP module in {type(layer).__name__}")


def get_input_device(model: nn.Module):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


@dataclass(frozen=True)
class ModuleSpec:
    layer: int
    kind: str = "resid"

    @classmethod
    def parse(cls, text: str) -> "ModuleSpec":
        text = text.strip()
        if ":" in text:
            left, kind = text.split(":", 1)
        else:
            left, kind = text, "resid"
        left = left.lstrip("Ll")
        kind = kind.lower()
        if kind not in {"resid", "attn", "mlp"}:
            raise ValueError(f"Unknown module kind: {kind}")
        return cls(layer=int(left), kind=kind)


def resolve_module(model: nn.Module, spec: ModuleSpec) -> nn.Module:
    layers = get_decoder_layers(model)
    layer = layers[spec.layer]
    if spec.kind == "resid":
        return layer
    if spec.kind == "attn":
        return get_attention_module(layer)
    return get_mlp_module(layer)


def get_attention_output_projection(layer: nn.Module) -> nn.Module:
    attn = get_attention_module(layer)
    for name in ("o_proj", "dense", "out_proj", "c_proj"):
        mod = getattr(attn, name, None)
        if mod is not None:
            return mod
    raise AttributeError(f"Could not locate attention output projection in {type(attn).__name__}")


def get_num_attention_heads(model: nn.Module, layer: nn.Module | None = None) -> int:
    cfg = getattr(model, "config", None)
    for name in ("num_attention_heads", "n_head", "num_heads"):
        v = getattr(cfg, name, None) if cfg is not None else None
        if v is not None:
            return int(v)
    if layer is not None:
        attn = get_attention_module(layer)
        for name in ("num_heads", "num_attention_heads", "n_head"):
            v = getattr(attn, name, None)
            if v is not None:
                return int(v)
    raise AttributeError("Could not infer number of attention heads")
