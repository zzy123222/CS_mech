from __future__ import annotations

import argparse, random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.interventions.patching import margin_with_hook
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import (get_decoder_layers, get_attention_output_projection,
                                     get_num_attention_heads, get_input_device)
from src.interventions.heads import _reshape_heads
from src.stats.bootstrap import bootstrap_mean
from src.utils.seed import set_seed


def _capture_heads(bundle, prompt, proj, n_heads, head_ids, pos):
    holder = {}
    def pre(_m, inputs):
        x = inputs[0]
        p = pos if pos >= 0 else x.shape[1] + pos
        holder["v"] = _reshape_heads(x, n_heads)[0, p, head_ids, :].detach().float().cpu()
    h = proj.register_forward_pre_hook(pre)
    try:
        enc = bundle.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = get_input_device(bundle.model)
        enc = {k: v.to(device) for k, v in enc.items()}
        bundle.model(**enc, use_cache=False)
    finally:
        h.remove()
    return holder["v"]


def _patch_prehook(clean_slices_cpu, head_ids, n_heads, pos):
    def pre(_m, inputs):
        x = inputs[0]
        p = pos if pos >= 0 else x.shape[1] + pos
        y = x.clone()
        yh = _reshape_heads(y, n_heads)
        yh[0, p, head_ids, :] = clean_slices_cpu.to(y.device, dtype=y.dtype)
        return (y,) + tuple(inputs[1:])
    return pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--heads", nargs="+", type=int, required=True)
    ap.add_argument("--max_pairs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    p_cfg = cfg.get("patching", {})
    clean_cond, corrupt_cond = p_cfg.get("clean_condition", "R-only"), p_cfg.get("corrupt_condition", "R+S-conflict")
    c = df[df.condition == clean_cond].set_index("group_id")
    x = df[df.condition == corrupt_cond].set_index("group_id")
    ids = list(c.index.intersection(x.index))
    random.Random(seed).shuffle(ids)
    ids = ids[: args.max_pairs]
    layer = get_decoder_layers(bundle.model)[args.layer]
    proj = get_attention_output_projection(layer)
    n_heads = get_num_attention_heads(bundle.model, layer)
    for h in args.heads:
        if h < 0 or h >= n_heads:
            raise SystemExit(f"head {h} outside [0,{n_heads})")
    head_ids = args.heads
    rows = []
    for gid in tqdm(ids, desc=f"L{args.layer}H{head_ids} patch"):
        clean, corrupt = c.loc[gid], x.loc[gid]
        cp = bundle.prepare_prompt(clean.prompt, enable_thinking=mcfg.get("enable_thinking"))
        xp = bundle.prepare_prompt(corrupt.prompt, enable_thinking=mcfg.get("enable_thinking"))
        r, s = str(corrupt.rule_answer), str(corrupt.shortcut_answer)
        cpos = len(bundle.tokenizer(cp, add_special_tokens=False)["input_ids"]) - 1
        xpos = len(bundle.tokenizer(xp, add_special_tokens=False)["input_ids"]) - 1
        slices = _capture_heads(bundle, cp, proj, n_heads, head_ids, cpos)
        m_clean = margin_with_hook(bundle, cp, r, s)
        m_corrupt = margin_with_hook(bundle, xp, r, s)
        handle = proj.register_forward_pre_hook(_patch_prehook(slices, head_ids, n_heads, xpos))
        try:
            m_patch = margin_with_hook(bundle, xp, r, s)
        finally:
            handle.remove()
        denom = m_clean - m_corrupt
        eps = 1e-8 if denom >= 0 else -1e-8
        rows.append({"group_id": gid, "layer": args.layer,
                     "heads": ",".join(map(str, head_ids)),
                     "clean_margin": m_clean, "corrupt_margin": m_corrupt,
                     "patched_margin": m_patch,
                     "recovery": (m_patch - m_corrupt) / (denom + eps)})
    out_dir = Path(cfg.get("output_dir", "results/p1_reasoning_switch")) / "heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    save_table(out, out_dir / f"head_patching_L{args.layer:03d}_H{'_'.join(map(str, head_ids))}.parquet")
    ok = out[out.recovery.abs() < 10]
    ci = bootstrap_mean(ok.recovery.values, n_boot=2000, seed=seed)
    print({"layer": args.layer, "heads": head_ids, "n": len(ok),
           "mean_recovery": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})


if __name__ == "__main__":
    main()
