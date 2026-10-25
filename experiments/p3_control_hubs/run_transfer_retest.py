from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.hooks import make_steer_hook
from src.interventions.patching import capture_module_vector
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed

L24ATTN = ModuleSpec(24, "attn")
TARGET_ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
NULL_ALPHAS = [-2.0, -1.0, 1.0, 2.0]
N_NULL = 20
N_TARGET = 64


def margin_rs(bundle, prompt, rule, short, hooks=None):
    z = score_answer_pairs(bundle, [prompt, prompt], [str(rule), str(short)],
                           batch_size=2, hooks=hooks)
    return float(z[0] - z[1])


def tpos(bundle, p):
    return len(bundle.tokenizer(p, add_special_tokens=False)["input_ids"]) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p3_control_hubs.yaml")
    ap.add_argument("--out", default="results/p3_transfer_retest")
    ap.add_argument("--n", type=int, default=N_TARGET)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 1))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    module = resolve_module(bundle.model, L24ATTN)

    def get(task, cond, split, n):
        g = df[(df.task_family == task) & (df.condition == cond) & (df.split == split)]
        return g.head(n) if len(g) >= n else g.sample(n, random_state=seed)

    rows = []
    for src_task, tgt_task in [("graph_reasoning", "retrieval"),
                               ("retrieval", "graph_reasoning")]:
        a = torch.stack([capture_module_vector(bundle,
                                               bundle.prepare_prompt(r.prompt, enable_thinking=False),
                                               L24ATTN, -1)
                         for _, r in tqdm(get(src_task, "R+S-conflict", "train", 32).iterrows(),
                                          desc=f"{src_task} conflict vecs", leave=False)])
        b = torch.stack([capture_module_vector(bundle,
                                               bundle.prepare_prompt(r.prompt, enable_thinking=False),
                                               L24ATTN, -1)
                         for _, r in tqdm(get(src_task, "R-only", "train", 32).iterrows(),
                                          desc=f"{src_task} ronly vecs", leave=False)])
        u = a.mean(0) - b.mean(0)
        scale = float(0.5 * (a.norm(dim=-1).mean() + b.norm(dim=-1).mean()))
        u = u / (u.norm() + 1e-8)
        u_scaled = u * scale

        tgt = get(tgt_task, "R+S-conflict", "test_ood", args.n)
        prompts = [(r.sample_id, bundle.prepare_prompt(r.prompt, enable_thinking=False),
                    r.rule_answer, r.shortcut_answer) for _, r in tgt.iterrows()]
        for sid, p, ra, sa in prompts:
            pos = tpos(bundle, p)
            m0 = margin_rs(bundle, p, ra, sa)
            for alpha in TARGET_ALPHAS:
                m = margin_rs(bundle, p, ra, sa,
                              [(module, make_steer_hook(u_scaled, alpha, pos))])
                rows.append({"src": src_task, "tgt": tgt_task, "direction": "task",
                             "alpha": alpha, "sid": sid, "effect": m - m0})
        rng = torch.Generator().manual_seed(seed + 7)
        for j in range(N_NULL):
            v = torch.randn(u.shape, generator=rng)
            v = v - (v @ u) * u
            v = v / (v.norm() + 1e-8) * scale
            for alpha in NULL_ALPHAS:
                for sid, p, ra, sa in prompts:
                    pos = tpos(bundle, p)
                    m0 = margin_rs(bundle, p, ra, sa)
                    m = margin_rs(bundle, p, ra, sa,
                                  [(module, make_steer_hook(v, alpha, pos))])
                    rows.append({"src": src_task, "tgt": tgt_task,
                                 "direction": f"null{j:02d}", "alpha": alpha,
                                 "sid": sid, "effect": m - m0})
        pd.DataFrame(rows).to_parquet(out_dir / "transfer_retest.parquet")
        print(f"[flush] {src_task}->{tgt_task} done")

    out = pd.DataFrame(rows)
    out.to_parquet(out_dir / "transfer_retest.parquet")
    summary = []
    for (s, t), g in out.groupby(["src", "tgt"]):
        task_g = g[g.direction == "task"]
        null_g = g[g.direction.str.startswith("null")]
        entry = {"src": s, "tgt": t, "n": int(task_g[task_g.alpha == 1.0].sid.nunique())}
        for alpha in TARGET_ALPHAS:
            te = task_g[task_g.alpha == alpha].effect
            ne = null_g[null_g.alpha.isin([alpha, -alpha])].effect if alpha in NULL_ALPHAS else null_g.effect
            q95 = float(np.percentile(ne, 95)) if len(ne) else np.nan
            entry[f"effect_a{alpha:+g}"] = float(te.mean())
            entry[f"null95_a{alpha:+g}"] = q95
            entry[f"exceeds_a{alpha:+g}"] = bool(abs(te.mean()) > abs(q95))
        for a in (1.0, 2.0):
            ep = float(task_g[task_g.alpha == a].effect.mean())
            em = float(task_g[task_g.alpha == -a].effect.mean())
            entry[f"antisym_a{a:g}"] = bool(ep * em < 0 and
                                            min(abs(ep), abs(em)) > 0.5 * max(abs(ep), abs(em)))
        entry["directional_transfer"] = bool(
            all(entry.get(f"exceeds_a{a:+g}", False) for a in (1.0, -1.0))
            and entry.get("antisym_a1", False))
        summary.append(entry)
    sm = pd.DataFrame(summary)
    sm.to_csv(out_dir / "transfer_retest_summary.csv", index=False)
    print(sm.to_string(index=False))
    json.dump({"n_target": args.n, "n_null": N_NULL, "seed": seed,
               "alphas": TARGET_ALPHAS, "null_alphas": NULL_ALPHAS,
               "date": "2026-10-14"}, open(out_dir / "run_metadata.json", "w"), indent=1)


if __name__ == "__main__":
    main()
