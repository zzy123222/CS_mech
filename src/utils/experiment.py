from __future__ import annotations

from pathlib import Path

from src.models.load import load_causal_lm
from src.utils.io import load_yaml


def load_bundle_from_model_config(path: str):
    cfg=load_yaml(path)
    bundle=load_causal_lm(
        cfg["model_id"], revision=cfg.get("revision"), dtype=cfg.get("dtype","bfloat16"),
        device_map=cfg.get("device_map","auto"), trust_remote_code=cfg.get("trust_remote_code",False),
        chat=cfg.get("chat",True), attn_implementation=cfg.get("attn_implementation")
    )
    return bundle,cfg


def prepare_prompts(bundle, prompts, enable_thinking=None):
    return [bundle.prepare_prompt(str(p), enable_thinking=enable_thinking) for p in prompts]
