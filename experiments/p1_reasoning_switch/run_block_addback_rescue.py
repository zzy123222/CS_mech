from __future__ import annotations

import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.patching import capture_module_vector, margin_with_hook
from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook, make_patch_hook
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.landmarks import landmark_token_position
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed
from src.stats.bootstrap import bootstrap_mean


def margin(bundle, p, r, s, hooks=None):
    return margin_with_hook(bundle, p, r, s, hooks=hooks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--module", default="L24:attn")
    ap.add_argument("--clean_condition", default="R+S-consistent",
                    help="structurally identical clean source (cue direction differs only)")
    ap.add_argument("--corrupt_condition", default="R+S-conflict")
    ap.add_argument("--position", default="last")
    ap.add_argument("--max_pairs", type=int, default=96)
    ap.add_argument("--controls", nargs="+", default=["same_answer", "wrong", "shuffled"],
                    choices=["same_answer", "wrong", "shuffled"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    spec = ModuleSpec.parse(args.module)
    module = resolve_module(bundle.model, spec)

    cl = df[df.condition == args.clean_condition].set_index("group_id")
    co = df[df.condition == args.corrupt_condition].set_index("group_id")
    ids = list(cl.index.intersection(co.index))
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[: args.max_pairs]
    clean_rows = df[df.condition == args.clean_condition]
    pool_by_letter = {}
    for r_ in clean_rows.itertuples():
        pool_by_letter.setdefault(str(r_.rule_answer), []).append(r_)

    ctl_ids = ids[: min(len(ids), 96)]
    ctl_prompts = [bundle.prepare_prompt(cl.loc[g].prompt, enable_thinking=mcfg.get("enable_thinking")) for g in ctl_ids]
    mean_vec = estimate_mean_vector(bundle, ctl_prompts, spec, max_samples=len(ctl_prompts))

    rows = []
    for gi, gid in enumerate(tqdm(ids, desc=f"addback {args.module}")):
        c_row, x_row = cl.loc[gid], co.loc[gid]
        cp = bundle.prepare_prompt(c_row.prompt, enable_thinking=mcfg.get("enable_thinking"))
        xp = bundle.prepare_prompt(x_row.prompt, enable_thinking=mcfg.get("enable_thinking"))
        r, s = str(x_row.rule_answer), str(x_row.shortcut_answer)
        cpos = landmark_token_position(bundle.tokenizer, cp, c_row.prompt, args.position)
        xpos = landmark_token_position(bundle.tokenizer, xp, x_row.prompt, args.position)

        a_clean = capture_module_vector(bundle, cp, spec, cpos)
        donors = {}
        if "same_answer" in args.controls:
            cands = [t for t in pool_by_letter.get(str(x_row.rule_answer), []) if t.group_id != gid]
            d = rng.choice(cands) if cands else rng.choice([t for t in clean_rows.itertuples() if t.group_id != gid])
            donors["same_answer"] = bundle.prepare_prompt(d.prompt, enable_thinking=mcfg.get("enable_thinking")), d.prompt
        if "wrong" in args.controls:
            d = rng.choice([t for t in clean_rows.itertuples() if t.group_id != gid])
            donors["wrong"] = bundle.prepare_prompt(d.prompt, enable_thinking=mcfg.get("enable_thinking")), d.prompt

        m_base = margin(bundle, xp, r, s)
        ko_hook = (module, make_mean_ablation_hook(mean_vec, xpos))
        m_ko = margin(bundle, xp, r, s, [ko_hook])
        denom = m_base - m_ko
        eps = 1e-8 if denom >= 0 else -1e-8
        rec = {"group_id": gid, "module": args.module, "m_base": m_base, "m_ko": m_ko, "denom": denom}

        m_res = margin(bundle, xp, r, s,
                       [ko_hook, (module, make_patch_hook(a_clean, target_pos=xpos))])
        rec["rescue_matched"] = (m_res - m_ko) / (denom + eps)

        for kind, (dp, draw) in donors.items():
            dpos = landmark_token_position(bundle.tokenizer, dp, draw, args.position)
            a_d = capture_module_vector(bundle, dp, spec, dpos)
            m_d = margin(bundle, xp, r, s,
                         [ko_hook, (module, make_patch_hook(a_d, target_pos=xpos))])
            rec[f"rescue_{kind}"] = (m_d - m_ko) / (denom + eps)
        if "shuffled" in args.controls:
            perm = torch.randperm(a_clean.numel(), generator=torch.Generator().manual_seed(seed * 100003 + gi))
            a_s = a_clean.reshape(-1)[perm].reshape(a_clean.shape)
            m_s = margin(bundle, xp, r, s,
                         [ko_hook, (module, make_patch_hook(a_s, target_pos=xpos))])
            rec["rescue_shuffled"] = (m_s - m_ko) / (denom + eps)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_dir = Path(cfg.get("output_dir", "results/p1_reasoning_switch")) / "rescue"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"addback_{args.module.replace(':', '_')}_{args.clean_condition.replace('+', '')}.parquet"
    save_table(out, out_dir / name)
    ok = out[out.denom.abs() >= 0.25]
    summary = []
    for col in [c for c in ok.columns if c.startswith("rescue_")]:
        ci = bootstrap_mean(ok[col].values, n_boot=2000, seed=seed)
        summary.append({"control": col.replace("rescue_", ""), "mean": ci["mean"],
                        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": len(ok)})
    sm = pd.DataFrame(summary)
    save_table(sm, out_dir / (name.replace(".parquet", "_summary.csv")))
    print(sm.to_string(index=False))
    if "rescue_matched" in ok.columns:
        m = float(ok.rescue_matched.mean())
        beat = all(m > float(ok[c].mean()) + 0.05 for c in ok.columns
                   if c.startswith("rescue_") and c != "rescue_matched")
        print(f"\nblock rescue fraction = {m:.3f}; >=0.30: {m >= 0.30}; exceeds all controls: {beat}")


if __name__ == "__main__":
    main()
