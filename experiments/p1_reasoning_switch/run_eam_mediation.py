from __future__ import annotations

import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.patching import capture_module_vector, margin_with_hook
from src.interventions.hooks import make_projection_ablation_hook, make_patch_hook
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.landmarks import landmark_token_position
from src.utils.module_access import (get_decoder_layers, get_attention_output_projection,
                                     get_num_attention_heads, ModuleSpec, resolve_module)
from src.interventions.heads import _reshape_heads
from src.stats.bootstrap import bootstrap_mean
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--e_layer", type=int, default=22, help="layer of E-direction removal")
    ap.add_argument("--a_layer", type=int, default=24, help="arbitration layer")
    ap.add_argument("--heads", nargs="+", type=int, default=[29, 31, 28])
    ap.add_argument("--vector", required=True, help="matched_s direction .npy at e_layer")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--clean_condition", default="R+S-consistent")
    ap.add_argument("--corrupt_condition", default="R+S-conflict")
    ap.add_argument("--max_pairs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    v = torch.from_numpy(np.load(args.vector)).float()
    e_module = resolve_module(bundle.model, ModuleSpec(args.e_layer, "resid"))
    a_module = resolve_module(bundle.model, ModuleSpec(args.a_layer, "attn"))
    proj = get_attention_output_projection(get_decoder_layers(bundle.model)[args.a_layer])
    n_heads = get_num_attention_heads(bundle.model, args.a_layer)

    cl = df[df.condition == args.clean_condition].set_index("group_id")
    co = df[df.condition == args.corrupt_condition].set_index("group_id")
    ids = list(cl.index.intersection(co.index))
    random.Random(seed).shuffle(ids)
    ids = ids[: args.max_pairs]

    rows = []
    for gid in tqdm(ids, desc="E->A->M"):
        c_row, x_row = cl.loc[gid], co.loc[gid]
        cp = bundle.prepare_prompt(c_row.prompt, enable_thinking=mcfg.get("enable_thinking"))
        xp = bundle.prepare_prompt(x_row.prompt, enable_thinking=mcfg.get("enable_thinking"))
        r, s = str(x_row.rule_answer), str(x_row.shortcut_answer)
        xpos = len(bundle.tokenizer(xp, add_special_tokens=False)["input_ids"]) - 1

        def run_with_head_capture(extra_hooks=None):
            holder = {}
            def pre(_m, inputs):
                x = inputs[0]
                holder["v"] = _reshape_heads(x, n_heads)[0, xpos, args.heads, :].detach().float().cpu()
            h = proj.register_forward_pre_hook(pre)
            try:
                m = margin_with_hook(bundle, xp, r, s, extra_hooks)
            finally:
                h.remove()
            return m, holder["v"]

        m_base, a_base = run_with_head_capture()

        e_hook = (e_module, make_projection_ablation_hook(v, alpha=args.alpha, target_pos=xpos))
        m_removed, a_removed = run_with_head_capture([e_hook])

        a_clean = capture_module_vector(bundle, cp, ModuleSpec(args.a_layer, "attn"),
                                        landmark_token_position(bundle.tokenizer, cp, c_row.prompt, "last"))
        m_rescue = margin_with_hook(bundle, xp, r, s,
                                    [e_hook, (a_module, make_patch_hook(a_clean, target_pos=xpos))])

        dA = (a_removed - a_base).norm().item()
        a_norm = a_base.norm().item() + 1e-8
        rows.append({"group_id": gid,
                     "m_base": m_base, "m_removed": m_removed, "m_rescue": m_rescue,
                     "dM": m_removed - m_base,
                     "rescue_fraction": (m_rescue - m_removed) / (m_base - m_removed + 1e-8),
                     "dA_norm": dA, "dA_rel": dA / a_norm,
                     "cos_A": float(torch.nn.functional.cosine_similarity(
                         (a_removed - a_base).flatten(), (a_base - a_removed).flatten(), dim=0) * -1)})

    out = pd.DataFrame(rows)
    out_dir = Path(cfg.get("output_dir", "results/p1_reasoning_switch")) / "mediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(out, out_dir / f"eam_L{args.e_layer}_to_L{args.a_layer}.parquet")
    agg = float((out.m_rescue - out.m_removed).sum() / ((out.m_base - out.m_removed).sum() + 1e-8))
    big = out[(out.m_base - out.m_removed).abs() >= 0.25]
    med_frac = float(big.rescue_fraction.median()) if len(big) else float("nan")
    cols = ["dM", "dA_rel"]
    summ = []
    for c in cols:
        ci = bootstrap_mean(out[c].values, n_boot=2000, seed=seed)
        summ.append({"metric": c, "mean": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": len(out)})
    summ.append({"metric": "rescue_fraction_aggregate", "mean": agg, "ci_low": float("nan"),
                 "ci_high": float("nan"), "n": len(out)})
    summ.append({"metric": "rescue_fraction_median_filtered", "mean": med_frac, "ci_low": float("nan"),
                 "ci_high": float("nan"), "n": len(big)})
    sm = pd.DataFrame(summ)
    save_table(sm, out_dir / f"eam_L{args.e_layer}_to_L{args.a_layer}_summary.csv")
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
