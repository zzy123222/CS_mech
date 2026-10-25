from __future__ import annotations

import sys, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.generate_error_monitor import build_check_prompt
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import (get_decoder_layers, get_attention_output_projection,
                                     get_num_attention_heads, resolve_module, ModuleSpec)
from src.interventions.heads import estimate_head_mean, make_head_mean_ablation_prehook
from src.models.scoring import score_answer_pairs
from src.hooks.core import first_tensor
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2_error_monitor.yaml")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--readout_layer", type=int, default=35)
    ap.add_argument("--max_samples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(args.seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"]).head(args.max_samples)

    layer = get_decoder_layers(bundle.model)[args.layer]
    proj = get_attention_output_projection(layer)
    n_heads = get_num_attention_heads(bundle.model, layer)
    readout = resolve_module(bundle.model, ModuleSpec(args.readout_layer, "resid"))

    edir_files = sorted(glob.glob(str(ROOT / "results/p2_error_monitor/analysis/error_direction_L035_t*.npy")))
    if not edir_files:
        raise SystemExit("No error direction files found")
    edir = torch.from_numpy(np.load(edir_files[0])).float()

    rows = []
    for head in range(n_heads):
        prompts = []
        for _, row in df.iterrows():
            t = max(1, min(int(row.n_steps), int(row.error_step) + 1))
            raw = build_check_prompt(row, "error", t)
            prompts.append(bundle.prepare_prompt(raw, enable_thinking=mcfg.get("enable_thinking")))

        clean_prompts = []
        for _, row in df.iterrows():
            t = max(1, min(int(row.n_steps), int(row.error_step)))
            raw = build_check_prompt(row, "clean", t)
            clean_prompts.append(bundle.prepare_prompt(raw, enable_thinking=mcfg.get("enable_thinking")))

        mean_head = estimate_head_mean(bundle, clean_prompts[:32], args.layer, head, max_samples=32)

        effects = []
        for prompt in prompts:
            pos = len(bundle.tokenizer(prompt, add_special_tokens=False)["input_ids"]) - 1
            holder = {}
            def readout_hook(_m, _i, output):
                h = first_tensor(output)
                holder["h"] = h[:, pos, :].detach().float().cpu()[0]

            h1 = readout.register_forward_hook(readout_hook)
            try:
                score_answer_pairs(bundle, [prompt], ["CHECK"], batch_size=1)
            finally:
                h1.remove()
            e_base = float(torch.dot(holder["h"], edir))

            pre = make_head_mean_ablation_prehook(mean_head, head, n_heads, pos)
            h2 = proj.register_forward_pre_hook(pre)
            holder.clear()
            h3 = readout.register_forward_hook(readout_hook)
            try:
                score_answer_pairs(bundle, [prompt], ["CHECK"], batch_size=1)
            finally:
                h2.remove()
                h3.remove()
            e_abl = float(torch.dot(holder["h"], edir))
            effects.append(e_abl - e_base)

        arr = np.array(effects)
        rows.append({"head": head, "delta_E": float(arr.mean()),
                     "sem": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0,
                     "n": len(arr)})
        if head % 8 == 7:
            print(f"  L{args.layer} H{head}: dE={arr.mean():.3f}±{arr.std(ddof=1)/np.sqrt(len(arr)):.3f}")

    out = pd.DataFrame(rows).sort_values("delta_E")
    out_dir = ROOT / "results/p2_error_monitor/heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"L{args.layer:03d}_head_screen.csv", index=False)
    print(f"\nTop-8 (largest dE drop = monitor-carrying heads):")
    print(out.head(8).to_string(index=False))
    print(f"\nRest {len(out)-8} heads median: {out.iloc[8:]['delta_E'].median():.4f}")


if __name__ == "__main__":
    main()
