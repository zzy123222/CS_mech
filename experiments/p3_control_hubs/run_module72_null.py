from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table
from src.utils.module_access import ModuleSpec, get_decoder_layers, resolve_module
from src.utils.seed import set_seed


def margin_rs(bundle, p, r, s, hooks=None):
    z = score_answer_pairs(bundle, [p, p], [str(r), str(s)], batch_size=2, hooks=hooks)
    return float(z[0] - z[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p3_control_hubs.yaml")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tasks", nargs="+", default=["graph_reasoning", "retrieval"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 1)))
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    n_layers = len(get_decoder_layers(bundle.model))
    rows = []
    for task in args.tasks:
        clean = df[(df.task_family == task) & (df.condition == "R-only")].head(args.n)
        conf = df[(df.task_family == task) & (df.condition == "R+S-conflict")].head(args.n)
        clean_prompts = [bundle.prepare_prompt(r.prompt, enable_thinking=False)
                         for _, r in clean.iterrows()]
        targets = [(r.sample_id, bundle.prepare_prompt(r.prompt, enable_thinking=False),
                    r.rule_answer, r.shortcut_answer) for _, r in conf.iterrows()]
        for lid in range(n_layers):
            for kind in ("attn", "mlp"):
                spec = ModuleSpec(lid, kind)
                module = resolve_module(bundle.model, spec)
                mv = estimate_mean_vector(bundle, clean_prompts, spec, max_samples=len(clean_prompts))
                for sid, p, ra, sa in targets:
                    pos = len(bundle.tokenizer(p, add_special_tokens=False)["input_ids"]) - 1
                    m0 = margin_rs(bundle, p, ra, sa)
                    m1 = margin_rs(bundle, p, ra, sa, [(module, make_mean_ablation_hook(mv, pos))])
                    rows.append({"task": task, "layer": lid, "kind": kind, "sid": sid,
                                 "m_base": m0, "damage": m0 - m1})
                pd.DataFrame(rows).to_csv("results/p3_control_hubs/module72_null_raw.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv("results/p3_control_hubs/module72_null_raw.csv", index=False)
    summary = []
    for task, g in out.groupby("task"):
        dmg = g.groupby(["layer", "kind"]).damage.mean().sort_values(ascending=False)
        l24 = dmg.loc[(24, "attn")]
        others = dmg.drop((24, "attn"))
        summary.append({"task": task, "L24attn_damage": float(l24),
                        "rank": int((others > l24).sum()) + 1,
                        "n_modules": len(dmg),
                        "exhaustive_p": float((others >= l24).sum() + 1) / len(others + 1),
                        "null_mean": float(others.mean()), "null_max": float(others.max()),
                        "null_p95": float(others.quantile(0.95))})
    sm = pd.DataFrame(summary)
    sm.to_csv("results/p3_control_hubs/module72_null.csv", index=False)
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
