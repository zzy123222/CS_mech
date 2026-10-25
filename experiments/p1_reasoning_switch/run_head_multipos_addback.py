from __future__ import annotations

import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.patching import margin_with_hook
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.landmarks import landmark_token_position
from src.utils.module_access import (get_decoder_layers, get_attention_output_projection,
                                     get_num_attention_heads, resolve_module, ModuleSpec)
from src.interventions.heads import _reshape_heads
from src.interventions.hooks import make_patch_hook
from src.stats.bootstrap import bootstrap_mean
from src.utils.seed import set_seed


def capture_heads_full(bundle, prompt, proj, n_heads, head_ids):
    holder = {}
    def pre(_m, inputs):
        x = inputs[0]
        holder["v"] = _reshape_heads(x, n_heads)[0][:, head_ids, :].detach().float().cpu()
    h = proj.register_forward_pre_hook(pre)
    try:
        enc = bundle.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(next(bundle.model.parameters()).device) for k, v in enc.items()}
        bundle.model(**enc, use_cache=False)
    finally:
        h.remove()
    return holder["v"]


def make_multipos_head_patch(clean_full, head_ids, n_heads, positions):
    def pre(_m, inputs):
        x = inputs[0]
        y = x.clone()
        yh = _reshape_heads(y, n_heads)
        pos = positions if positions is not None else list(range(x.shape[1]))
        for p in pos:
            if p < x.shape[1] and p < clean_full.shape[0]:
                yh[0, p, head_ids, :] = clean_full[p].to(y.device, dtype=y.dtype)
        return (y,) + tuple(inputs[1:])
    return pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--heads", nargs="+", type=int, default=[29, 31, 28])
    ap.add_argument("--clean_condition", default="R+S-consistent")
    ap.add_argument("--corrupt_condition", default="R+S-conflict")
    ap.add_argument("--max_pairs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    cl = df[df.condition == args.clean_condition].set_index("group_id")
    co = df[df.condition == args.corrupt_condition].set_index("group_id")
    ids = list(cl.index.intersection(co.index))
    random.Random(seed).shuffle(ids)
    ids = ids[: args.max_pairs]
    layer = get_decoder_layers(bundle.model)[args.layer]
    proj = get_attention_output_projection(layer)
    n_heads = get_num_attention_heads(bundle.model, layer)
    attn_module = resolve_module(bundle.model, ModuleSpec(args.layer, "attn"))
    head_ids = args.heads

    rows = []
    for gid in tqdm(ids, desc=f"B1-B4 L{args.layer}H{head_ids}"):
        c_row, x_row = cl.loc[gid], co.loc[gid]
        cp = bundle.prepare_prompt(c_row.prompt, enable_thinking=mcfg.get("enable_thinking"))
        xp = bundle.prepare_prompt(x_row.prompt, enable_thinking=mcfg.get("enable_thinking"))
        r, s = str(x_row.rule_answer), str(x_row.shortcut_answer)
        seq = len(bundle.tokenizer(xp, add_special_tokens=False)["input_ids"])
        clean_len = len(bundle.tokenizer(cp, add_special_tokens=False)["input_ids"])
        if clean_len != seq:
            continue
        last = seq - 1
        cue = landmark_token_position(bundle.tokenizer, xp, x_row.prompt, "cue_end")
        opt = landmark_token_position(bundle.tokenizer, xp, x_row.prompt, "options_end")

        clean_heads = capture_heads_full(bundle, cp, proj, n_heads, head_ids)
        blk = {}
        def blk_hook(_m, _i, output):
            from src.hooks.core import first_tensor
            blk["v"] = first_tensor(output).detach().float().cpu()[0]
        hb = attn_module.register_forward_hook(blk_hook)
        try:
            enc = bundle.tokenizer(cp, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(next(bundle.model.parameters()).device) for k, v in enc.items()}
            bundle.model(**enc, use_cache=False)
        finally:
            hb.remove()

        m_clean = margin_with_hook(bundle, cp, r, s)
        m_corr = margin_with_hook(bundle, xp, r, s)
        denom = m_clean - m_corr
        eps = 1e-8 if denom >= 0 else -1e-8
        rec = {"group_id": gid, "m_clean": m_clean, "m_conflict": m_corr, "denom": denom}

        for name, pos in [("B1_last", [last]), ("B2_cue_opt_last", [cue, opt, last]), ("B3_all", None)]:
            pre = make_multipos_head_patch(clean_heads, head_ids, n_heads, pos)
            h = proj.register_forward_pre_hook(pre)
            try:
                m = margin_with_hook(bundle, xp, r, s)
            finally:
                h.remove()
            rec[f"recovery_{name}"] = (m - m_corr) / (denom + eps)

        def blk_patch_hook(module, _i, output):
            from src.hooks.core import first_tensor, replace_first_tensor
            h_ = first_tensor(output)
            h2 = h_.clone()
            L = min(h2.shape[1], blk["v"].shape[0])
            h2[0, :L] = blk["v"][:L].to(h_.device, dtype=h_.dtype)
            return replace_first_tensor(output, h2)
        hb2 = attn_module.register_forward_hook(blk_patch_hook)
        try:
            m = margin_with_hook(bundle, xp, r, s)
        finally:
            hb2.remove()
        rec["recovery_B4_block_all"] = (m - m_corr) / (denom + eps)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_dir = Path(cfg.get("output_dir", "results/p1_reasoning_switch")) / "rescue"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(out, out_dir / f"head_multipos_L{args.layer:03d}_H{'_'.join(map(str, head_ids))}.parquet")
    ok = out[out.denom.abs() >= 0.25]
    summary = []
    for col in [c for c in ok.columns if c.startswith("recovery_")]:
        ci = bootstrap_mean(ok[col].values, n_boot=2000, seed=seed)
        summary.append({"condition": col.replace("recovery_", ""), "mean": ci["mean"],
                        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": len(ok)})
    sm = pd.DataFrame(summary)
    save_table(sm, out_dir / f"head_multipos_L{args.layer:03d}_H{'_'.join(map(str, head_ids))}_summary.csv")
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
