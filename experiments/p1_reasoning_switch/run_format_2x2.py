from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.ablation import estimate_mean_vector
from src.interventions.heads import (estimate_head_mean,
                                     make_head_mean_ablation_prehook)
from src.interventions.hooks import make_mean_ablation_hook, make_patch_hook
from src.interventions.patching import capture_module_vector, margin_with_hook
from src.models.load import load_causal_lm
from src.models.scoring import score_answer_pairs
from src.utils.io import load_table
from src.utils.module_access import (ModuleSpec, get_attention_output_projection,
                                     get_decoder_layers, get_num_attention_heads,
                                     resolve_module)
from src.utils.seed import set_seed

L24 = 24
GROUP7 = [28, 29, 30, 31]
CTRL_HEADS = [5, 12, 20]
OUT = Path("results/p1_format_2x2")
OUT.mkdir(parents=True, exist_ok=True)


def prep(bundle, prompt):
    return bundle.prepare_prompt(prompt, enable_thinking=False)


def tpos(bundle, p):
    return len(bundle.tokenizer(p, add_special_tokens=False)["input_ids"]) - 1


def margin_rs(bundle, prompt, rule, short, hooks=None):
    z = score_answer_pairs(bundle, [prompt, prompt], [str(rule), str(short)],
                           batch_size=2, hooks=hooks)
    return float(z[0] - z[1])


def flush(frame_rows, name):
    pd.DataFrame(frame_rows).to_parquet(OUT / f"{name}.parquet")


def behaviour(bundle, df, n, cell):
    rows = []
    for cond, g in df.groupby("condition"):
        for _, r in tqdm(g.head(n).iterrows(), total=min(n, len(g)),
                         desc=f"{cell} behavior {cond}", leave=False):
            p = prep(bundle, r.prompt)
            m = margin_rs(bundle, p, r.rule_answer, r.shortcut_answer)
            rows.append({"cell": cell, "condition": cond, "sample_id": r.sample_id,
                         "margin": m, "rule_preferred": m > 0})
    return rows


def pairs_of(df, n, seed=1):
    c = df[df.condition == "R-only"].set_index("group_id")
    x = df[df.condition == "R+S-conflict"].set_index("group_id")
    ids = sorted(c.index.intersection(x.index))
    rng = random.Random(seed)
    ids = rng.sample(ids, min(n, len(ids)))
    return [(gid, c.loc[gid], x.loc[gid]) for gid in ids]


def patch_curve(bundle, df, n, cell, rows_curve):
    pairs = pairs_of(df, n)
    layers = get_decoder_layers(bundle.model)
    for gid, cln, cor in tqdm(pairs, desc=f"{cell} patch curve", leave=False):
        pc, px = prep(bundle, cln.prompt), prep(bundle, cor.prompt)
        m_clean = margin_rs(bundle, pc, cln.rule_answer, cln.shortcut_answer)
        m_cor = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
        pos_c, pos_x = tpos(bundle, pc), tpos(bundle, px)
        for lid in range(len(layers)):
            spec = ModuleSpec(lid, "resid")
            v = capture_module_vector(bundle, pc, spec, pos_c)
            m_patch = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer,
                                [(resolve_module(bundle.model, spec),
                                  make_patch_hook(v, target_pos=pos_x))])
            denom = m_clean - m_cor
            rows_curve.append({"cell": cell, "group_id": gid, "layer": lid,
                               "m_clean": m_clean, "m_corrupt": m_cor,
                               "m_patch": m_patch,
                               "recovery": (m_patch - m_cor) / denom if abs(denom) > 0.25 else np.nan})
    return pairs


def block_battery(bundle, df, pairs, cell, rows_block):
    spec = ModuleSpec(L24, "attn")
    module = resolve_module(bundle.model, spec)
    clean_prompts = [prep(bundle, c.prompt) for _, c, _ in pairs]
    mean_vec = estimate_mean_vector(bundle, clean_prompts, spec, max_samples=len(clean_prompts))

    by_group = df[df.condition == "R+S-consistent"].set_index("group_id")
    conflict_by_group = df[df.condition == "R+S-conflict"].set_index("group_id")
    rng = random.Random(7)

    for gid, cln, cor in tqdm(pairs, desc=f"{cell} block battery", leave=False):
        px = prep(bundle, cor.prompt)
        pos = tpos(bundle, px)
        m_base = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
        ko = (module, make_mean_ablation_hook(mean_vec, pos))
        m_ko = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer, [ko])

        v_matched = capture_module_vector(
            bundle, prep(bundle, by_group.loc[gid].prompt), spec, -1) \
            if gid in by_group.index else None
        sa_pool = [g for g in conflict_by_group.index if g != gid]
        sa_g = next(g for g in sa_pool
                    if str(conflict_by_group.loc[g].rule_answer) == str(cor.rule_answer)
                    and conflict_by_group.loc[g].prompt is not None)
        w_g = next(g for g in sa_pool
                   if str(conflict_by_group.loc[g].rule_answer) != str(cor.rule_answer))
        v_same = capture_module_vector(
            bundle, prep(bundle, conflict_by_group.loc[sa_g].prompt), spec, -1)
        v_wrong = capture_module_vector(
            bundle, prep(bundle, conflict_by_group.loc[w_g].prompt), spec, -1)
        perm = torch.randperm(v_matched.numel(),
                              generator=torch.Generator().manual_seed(int(hash(str(gid)) % 2**31)))
        v_shuf = v_matched.reshape(-1)[perm].reshape(v_matched.shape)

        for dname, v in [("matched", v_matched), ("same_answer", v_same),
                         ("wrong", v_wrong), ("shuffled", v_shuf)]:
            m_res = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer,
                              [ko, (module, make_patch_hook(v, target_pos=pos))])
            rows_block.append({"cell": cell, "group_id": gid, "kind": f"rescue_{dname}",
                               "m_base": m_base, "m_int": m_res, "m_ko": m_ko})
        rows_block.append({"cell": cell, "group_id": gid, "kind": "block_KO",
                           "m_base": m_base, "m_int": m_ko, "m_ko": m_ko})

    layer = get_decoder_layers(bundle.model)[L24]
    proj = get_attention_output_projection(layer)
    n_heads = get_num_attention_heads(bundle.model, layer)
    head_means = {h: estimate_head_mean(bundle, clean_prompts, L24, h,
                                        max_samples=len(clean_prompts))
                  for h in range(n_heads)}
    for h in range(n_heads):
        for gid, cln, cor in tqdm(pairs, desc=f"{cell} head {h}", leave=False):
            px = prep(bundle, cor.prompt)
            pos = tpos(bundle, px)
            m_base = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
            pre = make_head_mean_ablation_prehook(head_means[h], h, n_heads, pos)
            handle = proj.register_forward_pre_hook(pre)
            try:
                m_abl = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
            finally:
                handle.remove()
            rows_block.append({"cell": cell, "group_id": gid, "kind": f"head_{h}",
                               "m_base": m_base, "m_int": m_abl, "m_ko": np.nan})
    for gid, cln, cor in tqdm(pairs, desc=f"{cell} group7", leave=False):
        px = prep(bundle, cor.prompt)
        pos = tpos(bundle, px)
        m_base = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
        pres = [proj.register_forward_pre_hook(
            make_head_mean_ablation_prehook(head_means[h], h, n_heads, pos))
            for h in GROUP7]
        try:
            m_abl = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
        finally:
            for hd in pres:
                hd.remove()
        rows_block.append({"cell": cell, "group_id": gid, "kind": "group7_joint",
                           "m_base": m_base, "m_int": m_abl, "m_ko": np.nan})


def run_cell(bundle, df, cell, n, rows_beh, rows_curve, rows_block):
    rows_beh.extend(behaviour(bundle, df, n, cell))
    flush(rows_beh, "behavior")
    pairs = patch_curve(bundle, df, n, cell, rows_curve)
    flush(rows_curve, "patch_curve")
    block_battery(bundle, df, pairs, cell, rows_block)
    flush(rows_block, "block_heads")


def summarize():
    beh = pd.read_parquet(OUT / "behavior.parquet")
    curve = pd.read_parquet(OUT / "patch_curve.parquet")
    blk = pd.read_parquet(OUT / "block_heads.parquet")
    out = {}
    for cell, g in beh.groupby("cell"):
        ronly = g[g.condition == "R-only"].margin
        out.setdefault(cell, {})["behavior"] = {
            c: {"mean_margin": float(v.margin.mean()),
                "rule_pref": float(v.rule_preferred.mean())}
            for c, v in g.groupby("condition")}
        out[cell]["ronly_margin"] = float(ronly.mean())
    for cell, g in curve.groupby("cell"):
        ok = g.dropna(subset=["recovery"])
        lay = ok.groupby("layer").recovery.mean()
        out.setdefault(cell, {})["patch_L23"] = float(lay.get(23, np.nan))
        out[cell]["patch_L24"] = float(lay.get(24, np.nan))
    for cell, g in blk.groupby("cell"):
        ronly = out.get(cell, {}).get("ronly_margin", np.nan)
        sd = float(g[g.kind == "block_KO"].m_base.std() + 1e-9)
        d = {}
        ko = g[g.kind == "block_KO"]
        d["block_KO_nats"] = float((ko.m_base - ko.m_int).mean())
        d["block_KO_by_ronly"] = d["block_KO_nats"] / ronly if ronly else np.nan
        for k in ["matched", "same_answer", "wrong", "shuffled"]:
            r = g[g.kind == f"rescue_{k}"]
            num, den = (r.m_int - r.m_ko).mean(), (r.m_base - r.m_ko).mean()
            d[f"rescueROM_{k}"] = float(num / den) if den else np.nan
        heads = g[g.kind.str.startswith("head_")].copy()
        heads["head"] = heads.kind.str.slice(5).astype(int)
        heads["damage"] = heads.m_base - heads.m_int
        hs = heads.groupby("head").damage.mean()
        d["head_damage_top4"] = {int(h): round(float(v), 3)
                                 for h, v in hs.sort_values(ascending=False).head(4).items()}
        g7 = g[g.kind == "group7_joint"]
        d["group7_KO_nats"] = float((g7.m_base - g7.m_int).mean())
        d["group7_KO_by_ronly"] = d["group7_KO_nats"] / ronly if ronly else np.nan
        d["baseline_sd"] = sd
        out[cell]["block"] = d
    json.dump(out, open(OUT / "format_2x2_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/generated/p1_graph_4b_seed1.parquet")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--cells", nargs="+",
                    default=["base_raw", "base_chat", "inst_raw", "inst_chat"])
    ap.add_argument("--adapters", nargs="*", default=[],
                    help="like general_sft_seed1:1000 results/posttraining/.../checkpoint-xxx")
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args()
    if args.summarize_only:
        summarize()
        return
    set_seed(1)
    df = load_table(args.dataset)
    rows_beh, rows_curve, rows_block = [], [], []
    for name, rows in [("behavior", rows_beh), ("patch_curve", rows_curve),
                       ("block_heads", rows_block)]:
        p = OUT / f"{name}.parquet"
        if p.exists():
            old = pd.read_parquet(p).to_dict("records")
            rows.extend(old)
            print(f"resume: {name} reloaded {len(old)} rows")

    def cells_done():
        p = OUT / "block_heads.parquet"
        return set(pd.read_parquet(p).cell.unique()) if p.exists() else set()

    for model_id, mtag in [("Qwen/Qwen3-4B-Base", "base"), ("Qwen/Qwen3-4B", "inst")]:
        want = [c for c in args.cells if c.startswith(mtag)]
        if not want:
            continue
        if all(f"{mtag}_{fmt}" in cells_done() for fmt in ("raw", "chat")):
            print(f"skip {mtag}: both cells done")
            continue
        bundle = load_causal_lm(model_id, dtype="bfloat16", device_map="auto", chat=True)
        for fmt in ("raw", "chat"):
            cell = f"{mtag}_{fmt}"
            if cell not in want or cell in cells_done():
                continue
            bundle.chat = fmt == "chat"
            print(f"=== cell {cell} (chat={bundle.chat}) ===")
            run_cell(bundle, df, cell, args.n, rows_beh, rows_curve, rows_block)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()

    for spec_text in args.adapters:
        arm_dir, step = spec_text.rsplit(":", 1)
        cell = f"adapter_{arm_dir.split('/')[-1]}_s{step}_raw"
        if cell in cells_done():
            continue
        from peft import PeftModel
        ckpt = Path("posttraining/results/formal") / arm_dir / f"checkpoint-{step}"
        bundle = load_causal_lm("Qwen/Qwen3-4B-Base", dtype="bfloat16",
                                device_map="auto", chat=False)
        bundle.model = PeftModel.from_pretrained(bundle.model, str(ckpt))
        bundle.model.eval()
        print(f"=== cell {cell} ({ckpt}) ===")
        pairs = pairs_of(df, args.n)
        rows_beh.extend(behaviour(bundle, df, args.n, cell))
        flush(rows_beh, "behavior")
        spec = ModuleSpec(L24, "attn")
        module = resolve_module(bundle.model, spec)
        clean_prompts = [prep(bundle, c.prompt) for _, c, _ in pairs]
        mean_vec = estimate_mean_vector(bundle, clean_prompts, spec,
                                        max_samples=len(clean_prompts))
        layer = get_decoder_layers(bundle.model)[L24]
        proj = get_attention_output_projection(layer)
        n_heads = get_num_attention_heads(bundle.model, layer)
        for gid, cln, cor in tqdm(pairs, desc=f"{cell} block", leave=False):
            px = prep(bundle, cor.prompt)
            pos = tpos(bundle, px)
            m_base = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
            m_ko = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer,
                             [(module, make_mean_ablation_hook(mean_vec, pos))])
            rows_block.append({"cell": cell, "group_id": gid, "kind": "block_KO",
                               "m_base": m_base, "m_int": m_ko, "m_ko": m_ko})
        for h in GROUP7 + CTRL_HEADS:
            hm = estimate_head_mean(bundle, clean_prompts, L24, h,
                                    max_samples=len(clean_prompts))
            for gid, cln, cor in tqdm(pairs, desc=f"{cell} head {h}", leave=False):
                px = prep(bundle, cor.prompt)
                pos = tpos(bundle, px)
                m_base = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
                pre = make_head_mean_ablation_prehook(hm, h, n_heads, pos)
                handle = proj.register_forward_pre_hook(pre)
                try:
                    m_abl = margin_rs(bundle, px, cor.rule_answer, cor.shortcut_answer)
                finally:
                    handle.remove()
                rows_block.append({"cell": cell, "group_id": gid, "kind": f"head_{h}",
                                   "m_base": m_base, "m_int": m_abl, "m_ko": np.nan})
        flush(rows_block, "block_heads")
        del bundle
        gc.collect()
        torch.cuda.empty_cache()

    summarize()


if __name__ == "__main__":
    main()
