from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.heads import (estimate_head_mean,
                                     estimate_proj_slice_mean,
                                     make_head_mean_ablation_prehook,
                                     make_slice_mean_ablation_prehook,
                                     make_slice_patch_prehook)
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table
from src.utils.module_access import (get_attention_output_projection,
                                     get_decoder_layers,
                                     get_num_attention_heads)
from src.utils.seed import set_seed

L24, HEAD_DIM, N_KV = 24, 128, 8
QGROUP7 = [28, 29, 30, 31]
OUT = Path("results/p1_kv_group")


def margin_rs(bundle, prompt, rule, short, prehooks=()):
    handles = []
    try:
        for proj, pre in prehooks:
            handles.append(proj.register_forward_pre_hook(pre))
        z = score_answer_pairs(bundle, [prompt, prompt], [str(rule), str(short)],
                               batch_size=2)
        return float(z[0] - z[1])
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1_reasoning_switch_4b_seed1.yaml")
    ap.add_argument("--battery", default="data/generated/p3_task_battery.parquet")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tasks", nargs="+", default=["graph_reasoning", "retrieval"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 1))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    OUT.mkdir(parents=True, exist_ok=True)
    layer = get_decoder_layers(bundle.model)[L24]
    attn = layer.self_attn
    o_proj = get_attention_output_projection(layer)
    n_heads = get_num_attention_heads(bundle.model, layer)

    def prep(p):
        return bundle.prepare_prompt(p, enable_thinking=mcfg.get("enable_thinking"))

    def tpos(p):
        return len(bundle.tokenizer(p, add_special_tokens=False)["input_ids"]) - 1

    df = load_table(args.battery)
    rows = []
    for task in args.tasks:
        clean = df[(df.task_family == task) & (df.condition == "R-only")].head(args.n)
        clean_prompts = [prep(r.prompt) for _, r in clean.iterrows()]
        kv_means = {}
        for grp in range(N_KV):
            d0, d1 = grp * HEAD_DIM, (grp + 1) * HEAD_DIM
            kv_means[grp] = (estimate_proj_slice_mean(bundle, clean_prompts, L24,
                                                      "k_proj", d0, d1),
                             estimate_proj_slice_mean(bundle, clean_prompts, L24,
                                                      "v_proj", d0, d1))
        head_means = {h: estimate_head_mean(bundle, clean_prompts, L24, h,
                                           max_samples=len(clean_prompts))
                      for h in QGROUP7}

        conflict = df[(df.task_family == task) & (df.condition == "R+S-conflict")].head(args.n)
        consistent = df[(df.task_family == task) &
                        (df.condition == "R+S-consistent")].set_index("group_id")
        cgrp = conflict.set_index("group_id")

        for _, r in tqdm(conflict.iterrows(), total=len(conflict), desc=f"{task}", leave=False):
            p = prep(r.prompt)
            m_base = margin_rs(bundle, p, r.rule_answer, r.shortcut_answer)

            for grp in range(N_KV):
                mk, mv = kv_means[grp]
                d0, d1 = grp * HEAD_DIM, (grp + 1) * HEAD_DIM
                pre_k = make_slice_mean_ablation_prehook(mk, d0, d1)
                pre_v = make_slice_mean_ablation_prehook(mv, d0, d1)
                m = margin_rs(bundle, p, r.rule_answer, r.shortcut_answer,
                              [(attn.k_proj, pre_k), (attn.v_proj, pre_v)])
                rows.append({"task": task, "sid": r.sample_id, "kind": f"kvgroup_{grp}",
                             "m_base": m_base, "m_int": m})
            for name, hs in [("q_{28,29,31}", [28, 29, 31]),
                             ("q_{28,29,30,31}", QGROUP7),
                             ("q_{30}", [30])]:
                pres = [(o_proj, make_head_mean_ablation_prehook(head_means[h], h, n_heads, tpos(p)))
                        for h in hs]
                m = margin_rs(bundle, p, r.rule_answer, r.shortcut_answer, pres)
                rows.append({"task": task, "sid": r.sample_id, "kind": name,
                             "m_base": m_base, "m_int": m})

            if task == "graph_reasoning":
                d0, d1 = 7 * HEAD_DIM, 8 * HEAD_DIM
                mk, mv = kv_means[7]
                ko = [(attn.k_proj, make_slice_mean_ablation_prehook(mk, d0, d1)),
                      (attn.v_proj, make_slice_mean_ablation_prehook(mv, d0, d1))]
                m_ko = margin_rs(bundle, p, r.rule_answer, r.shortcut_answer, ko)
                donor = consistent.loc[r.group_id] if r.group_id in consistent.index else None
                if donor is not None:
                    dp = prep(donor.prompt)
                    dpos = tpos(dp)
                    def capture_donor_vec(prompt, proj_name, pos, d0=d0, d1=d1):
                        holder = {}
                        proj = getattr(attn, proj_name)
                        def pre(_m, inputs):
                            holder["v"] = inputs[0][0, pos, d0:d1].detach().float().cpu()
                        h = proj.register_forward_pre_hook(pre)
                        try:
                            enc = bundle.tokenizer(prompt, return_tensors="pt",
                                                   add_special_tokens=False)
                            enc = {k: v.to(next(bundle.model.parameters()).device)
                                   for k, v in enc.items()}
                            bundle.model(**enc, use_cache=False)
                        finally:
                            h.remove()
                        return holder["v"]
                    vk = capture_donor_vec(dp, "k_proj", dpos)
                    vv = capture_donor_vec(dp, "v_proj", dpos)
                    pos_idx = tpos(p)

                    def make_slice_terminal_patch(vec, dim0, dim1, pos):
                        def pre(_m, inputs):
                            x = inputs[0]
                            y = x.clone()
                            y[:, pos, dim0:dim1] = vec.to(y.device, dtype=y.dtype)
                            return (y,) + tuple(inputs[1:])
                        return pre
                    pk = make_slice_terminal_patch(vk, d0, d1, pos_idx)
                    pv = make_slice_terminal_patch(vv, d0, d1, pos_idx)
                    m_kv_patch = margin_rs(bundle, p, r.rule_answer, r.shortcut_answer,
                                           [(attn.k_proj, pk), (attn.v_proj, pv)])
                    rows.append({"task": task, "sid": r.sample_id,
                                 "kind": "kv7_patch_donor_terminal", "m_base": m_base,
                                 "m_int": m_kv_patch, "m_ko": m_ko})
                rows.append({"task": task, "sid": r.sample_id, "kind": "kv7_KO",
                             "m_base": m_base, "m_int": m_ko, "m_ko": m_ko})
        pd.DataFrame(rows).to_parquet(OUT / "kv_group.parquet")
        print(f"[flush] {task}: {len(rows)} rows")

    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "kv_group.parquet")
    summary = []
    for (task, kind), g in out.groupby(["task", "kind"]):
        dmg = (g.m_base - g.m_int)
        entry = {"task": task, "kind": kind, "n": len(g), "damage_nats": float(dmg.mean()),
                 "damage_sem": float(dmg.sem())}
        if "m_ko" in g and g.m_ko.notna().all() and kind != "kv7_KO":
            num = (g.m_int - g.m_ko).mean()
            den = (g.m_base - g.m_ko).mean()
            entry["rescue_ROM"] = float(num / den) if den else np.nan
        summary.append(entry)
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "kv_group_summary.csv", index=False)
    print(sm.to_string(index=False))
    json.dump({"n": args.n, "groups": N_KV, "head_dim": HEAD_DIM,
               "layer": L24, "date": "2026-10-14"},
              open(OUT / "run_metadata.json", "w"), indent=1)


if __name__ == "__main__":
    main()
