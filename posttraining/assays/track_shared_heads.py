from __future__ import annotations
import sys, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
import torch

from src.interventions.heads import estimate_head_mean, make_head_mean_ablation_prehook
from src.models.scoring import score_answer_pairs
from src.utils.io import load_table
from src.utils.module_access import (get_decoder_layers, get_attention_output_projection,
                                     get_num_attention_heads)
from src.utils.seed import set_seed

SHARED_HEADS = [29, 31, 28]
CONTROL_HEADS = [5, 12, 20]


def load_merged(base_id, adapter_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


class Bundle:
    def __init__(self, model, tok):
        self.model = model
        self.tokenizer = tok

    def prepare_prompt(self, prompt, enable_thinking=None):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def margin(bundle, p, r, s, prehook=None, proj=None):
    handle = proj.register_forward_pre_hook(prehook) if prehook else None
    try:
        z = score_answer_pairs(bundle, [p, p], [str(r), str(s)], batch_size=2)
        return float(z[0] - z[1])
    finally:
        if handle:
            handle.remove()


def scan_model(model, tok, df_graph, df_retrieval, tag, n, rows):
    bundle = Bundle(model, tok)
    layer = get_decoder_layers(model)[24]
    proj = get_attention_output_projection(layer)
    n_heads = get_num_attention_heads(model, layer)

    for task, df_task in [("graph", df_graph), ("retrieval", df_retrieval)]:
        clean = df_task[df_task.condition == "R-only"].head(n)
        conflict = df_task[df_task.condition == "R+S-conflict"].head(n)
        clean_prompts = [bundle.prepare_prompt(r.prompt, enable_thinking=False)
                         for _, r in clean.iterrows()]
        conf = [(bundle.prepare_prompt(r.prompt, enable_thinking=False),
                 r.rule_answer, r.shortcut_answer) for _, r in conflict.iterrows()]

        for head in SHARED_HEADS + CONTROL_HEADS:
            mean = estimate_head_mean(bundle, clean_prompts, 24, head,
                                      max_samples=len(clean_prompts))
            damages = []
            for p, ra, sa in conf:
                pos = len(bundle.tokenizer(p, add_special_tokens=False)["input_ids"]) - 1
                pre = make_head_mean_ablation_prehook(mean, head, n_heads, pos)
                base = margin(bundle, p, ra, sa)
                abl = margin(bundle, p, ra, sa, pre, proj)
                damages.append(base - abl)
            d = np.asarray(damages)
            boot = np.random.default_rng(42).integers(0, len(d), size=(2000, len(d)))
            bm = d[boot].mean(1)
            rows.append({
                "model": tag, "task": task, "layer": 24, "head": head,
                "is_shared": head in SHARED_HEADS,
                "damage_mean": float(d.mean()),
                "damage_ci_low": float(np.percentile(bm, 2.5)),
                "damage_ci_high": float(np.percentile(bm, 97.5)),
                "n": len(d),
            })
            print(f"  {tag} {task} H{head}: damage={d.mean():+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", type=int, default=[0, 200, 500, 1000])
    ap.add_argument("--arms", nargs="*", default=["general_sft_seed1", "correction_sft_seed1",
                                                   "sham_correction_seed1"])
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--include_instruct", action="store_true", default=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="posttraining/results/formal/head_tracking.csv")
    args = ap.parse_args()

    set_seed(1)
    battery = load_table("data/generated/p3_task_battery.parquet")
    df_graph = battery[battery.task_family == "graph_reasoning"]
    df_retrieval = battery[battery.task_family == "retrieval"]

    rows = []
    if args.include_instruct:
        model, tok = load_merged("Qwen/Qwen3-4B", None)
        scan_model(model, tok, df_graph, df_retrieval, "instruct_ref", args.n, rows)
        del model
        gc.collect(); torch.cuda.empty_cache()

    for arm in args.arms:
        for step in args.steps:
            ckpt = ROOT / "posttraining/results/formal" / arm / f"step_{step:04d}"
            if not ckpt.exists():
                print(f"[skip] {ckpt}")
                continue
            model, tok = load_merged(args.base_model, str(ckpt))
            scan_model(model, tok, df_graph, df_retrieval, f"{arm}@{step}", args.n, rows)
            del model
            gc.collect(); torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
