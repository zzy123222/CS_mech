from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.generate_error_monitor import build_check_prompt
from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import (make_mean_ablation_hook,
                                     make_projection_ablation_hook,
                                     make_steer_hook)
from src.interventions.patching import capture_module_vector
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed

RESID_SWEEP = [f"L{l}" for l in range(18, 27)]
MODULE_SCAN = [f"L{l}:{k}" for l in range(20, 27) for k in ("attn", "mlp")]
FACTORIAL = {"M": "L20", "C": "L22:mlp", "O": "L25:attn"}
STEER_ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
N_NULL_DIRS = 20


def margin(bundle, prompt, pos_ans, neg_ans, hooks=None):
    z = score_answer_pairs(bundle, [prompt, prompt], [pos_ans, neg_ans],
                           batch_size=2, hooks=hooks)
    return float(z[0] - z[1])


def auc(err, clean):
    e, c = np.asarray(err), np.asarray(clean)
    n1, n0 = len(e), len(c)
    ranks = pd.Series(np.concatenate([e, c])).rank().values
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sd_stats(err, clean):
    e, c = np.asarray(err, float), np.asarray(clean, float)
    sp = np.sqrt((e.var(ddof=1) + c.var(ddof=1)) / 2)
    d = (e.mean() - c.mean()) / (sp + 1e-9)
    k = -(e.mean() + c.mean()) / (2 * sp + 1e-9)
    return d, k


def paired_boot_disc(d_err, d_clean, n_boot=10000, seed=777):
    rng = np.random.default_rng(seed)
    d_err, d_clean = np.asarray(d_err), np.asarray(d_clean)
    n = len(d_err)
    idx = rng.integers(0, n, size=(n_boot, n))
    stat = d_err[idx].mean(1) - d_clean[idx].mean(1)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2_error_monitor.yaml")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default="results/p2_specificity")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 11))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    if len(df) > args.n:
        df = df.sample(args.n, random_state=seed).reset_index(drop=True)
    pos_ans = str(cfg.get("check_answer", "CHECK"))
    neg_ans = str(cfg.get("continue_answer", "CONTINUE"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = []
    for _, row in df.iterrows():
        t = max(1, min(int(row.n_steps), int(row.error_step) + 1))
        pe = bundle.prepare_prompt(build_check_prompt(row, "error", t),
                                   enable_thinking=mcfg.get("enable_thinking"))
        pc = bundle.prepare_prompt(build_check_prompt(row, "clean", t),
                                   enable_thinking=mcfg.get("enable_thinking"))
        epos = len(bundle.tokenizer(pe, add_special_tokens=False)["input_ids"]) - 1
        cpos = len(bundle.tokenizer(pc, add_special_tokens=False)["input_ids"]) - 1
        pre.append({"sample_id": row.sample_id, "t": t, "pe": pe, "pc": pc,
                    "epos": epos, "cpos": cpos})

    for r in tqdm(pre, desc="baseline"):
        r["C_err"] = margin(bundle, r["pe"], pos_ans, neg_ans)
        r["C_clean"] = margin(bundle, r["pc"], pos_ans, neg_ans)

    rows = []
    def record(sid, cond, intervention, alpha, m):
        rows.append({"sample_id": sid, "condition": cond, "intervention": intervention,
                     "alpha": alpha, "margin": m})

    for r in pre:
        record(r["sample_id"], "error", "baseline", np.nan, r["C_err"])
        record(r["sample_id"], "clean", "baseline", np.nan, r["C_clean"])

    def flush(tag):
        pd.DataFrame(rows).to_parquet(out_dir / "specificity_panel.parquet")
        print(f"[checkpoint:{tag}] {len(rows)} rows")

    specs = sorted(set(RESID_SWEEP + MODULE_SCAN + list(FACTORIAL.values())),
                   key=lambda s: ModuleSpec.parse(s).layer)
    mean_vecs = {}
    clean_prompts = [r["pc"] for r in pre]
    for s in specs:
        spec = ModuleSpec.parse(s)
        mean_vecs[s] = estimate_mean_vector(bundle, clean_prompts, spec,
                                            max_samples=len(clean_prompts))
    print("clean-mean vectors done:", list(mean_vecs))

    for s in specs:
        module = resolve_module(bundle.model, ModuleSpec.parse(s))
        for r in tqdm(pre, desc=f"ablate {s}", leave=False):
            he = (module, make_mean_ablation_hook(mean_vecs[s], r["epos"]))
            hc = (module, make_mean_ablation_hook(mean_vecs[s], r["cpos"]))
            record(r["sample_id"], "error", f"ablate:{s}", np.nan,
                   margin(bundle, r["pe"], pos_ans, neg_ans, [he]))
            record(r["sample_id"], "clean", f"ablate:{s}", np.nan,
                   margin(bundle, r["pc"], pos_ans, neg_ans, [hc]))
        flush(f"ablate:{s}")

    mods = {k: resolve_module(bundle.model, ModuleSpec.parse(v))
            for k, v in FACTORIAL.items()}
    for combo in range(8):
        keys = [k for k in ("M", "C", "O") if combo & (1 << ("MCO".index(k)))]
        name = "factorial:" + ("".join(keys) if keys else "none")
        for r in tqdm(pre, desc=name, leave=False):
            he = [(mods[k], make_mean_ablation_hook(mean_vecs[FACTORIAL[k]], r["epos"]))
                  for k in keys]
            hc = [(mods[k], make_mean_ablation_hook(mean_vecs[FACTORIAL[k]], r["cpos"]))
                  for k in keys]
            record(r["sample_id"], "error", name, np.nan,
                   margin(bundle, r["pe"], pos_ans, neg_ans, he))
            record(r["sample_id"], "clean", name, np.nan,
                   margin(bundle, r["pc"], pos_ans, neg_ans, hc))
        flush(name)

    def centered_direction(spec_text):
        spec = ModuleSpec.parse(spec_text)
        ve = torch.stack([capture_module_vector(bundle, r["pe"], spec, r["epos"])
                          for r in pre])
        vc = torch.stack([capture_module_vector(bundle, r["pc"], spec, r["cpos"])
                          for r in pre])
        u = (ve.mean(0) - vc.mean(0))
        return u / (u.norm() + 1e-8), float(ve.norm(dim=-1).mean()), float(vc.norm(dim=-1).mean())

    u25, n25e, n25c = centered_direction("L25:attn")
    u20, _, _ = centered_direction("L20")
    torch.save(u25, out_dir / "direction_L25attn_centered.pt")
    torch.save(u20, out_dir / "direction_L20resid_centered.pt")
    alpha_scale = 0.5 * (n25e + n25c)
    print(f"direction norms L25:attn err {n25e:.1f} clean {n25c:.1f}")

    steer_mod = resolve_module(bundle.model, ModuleSpec.parse("L25:attn"))
    rng = torch.Generator().manual_seed(seed)
    null_dirs = []
    while len(null_dirs) < N_NULL_DIRS:
        v = torch.randn(u25.shape, generator=rng)
        v = v - (v @ u25) * u25
        v = v / (v.norm() + 1e-8)
        null_dirs.append(v)

    def steer_rows(name, vec, alphas):
        for a in alphas:
            scaled = vec * alpha_scale
            for r in tqdm(pre, desc=f"steer {name} a={a}", leave=False):
                he = (steer_mod, make_steer_hook(scaled, a, r["epos"]))
                hc = (steer_mod, make_steer_hook(scaled, a, r["cpos"]))
                record(r["sample_id"], "error", name, a,
                       margin(bundle, r["pe"], pos_ans, neg_ans, [he]))
                record(r["sample_id"], "clean", name, a,
                       margin(bundle, r["pc"], pos_ans, neg_ans, [hc]))
            flush(f"{name}:{a}")

    steer_rows("steer:L25attn", u25, STEER_ALPHAS)
    for j, v in enumerate(null_dirs):
        steer_rows(f"steer:null{j:02d}", v, [-1.0, 1.0])

    proj_mod = resolve_module(bundle.model, ModuleSpec.parse("L20"))
    for r in tqdm(pre, desc="L20 projection removal", leave=False):
        he = (proj_mod, make_projection_ablation_hook(u20, 1.0, r["epos"]))
        hc = (proj_mod, make_projection_ablation_hook(u20, 1.0, r["cpos"]))
        record(r["sample_id"], "error", "projrm:L20", np.nan,
               margin(bundle, r["pe"], pos_ans, neg_ans, [he]))
        record(r["sample_id"], "clean", "projrm:L20", np.nan,
               margin(bundle, r["pc"], pos_ans, neg_ans, [hc]))
    flush("projrm")

    panel = pd.DataFrame(rows)
    panel.to_parquet(out_dir / "specificity_panel.parquet")
    base = {(r["sample_id"], c): r[f"C_{c}"] for r in pre for c in ("error", "clean")}
    summary = []
    for intervention, grp in panel[panel.intervention != "baseline"].groupby(
            ["intervention", "alpha"], dropna=False):
        d = {}
        for cond in ("error", "clean"):
            g = grp[grp.condition == cond].set_index("sample_id").margin
            deltas = np.array([g.loc[s] - base[(s, cond)] for s in g.index])
            d[f"delta_{cond}"] = float(deltas.mean())
            d[f"n_{cond}"] = int(len(deltas))
            d[f"post_{cond}"] = g.mean()
        d["delta_disc"] = d["delta_error"] - d["delta_clean"]
        lo, hi = paired_boot_disc(
            grp[grp.condition == "error"].set_index("sample_id").margin.values
            - np.array([base[(s, "error")] for s in grp[grp.condition == "error"].sample_id]),
            grp[grp.condition == "clean"].set_index("sample_id").margin.values
            - np.array([base[(s, "clean")] for s in grp[grp.condition == "clean"].sample_id]))
        d["delta_disc_ci_low"], d["delta_disc_ci_high"] = lo, hi
        ge = grp[grp.condition == "error"].set_index("sample_id").margin
        gc = grp[grp.condition == "clean"].set_index("sample_id").margin
        common = ge.index.intersection(gc.index)
        e = ge.loc[common].values
        c = gc.loc[common].values
        be = np.array([base[(s, "error")] for s in common])
        bc = np.array([base[(s, "clean")] for s in common])
        d["auc_pre"], d["auc_post"] = auc(be, bc), auc(e, c)
        d["dprime_pre"], d["crit_pre"] = sd_stats(be, bc)
        d["dprime_post"], d["crit_post"] = sd_stats(e, c)
        d["intervention"], d["alpha"] = intervention
        summary.append(d)
    sm = pd.DataFrame(summary)
    sm.to_csv(out_dir / "specificity_summary.csv", index=False)
    print(sm[sm.alpha.isna()].to_string())

    meta = {
        "model_config": cfg["model_config"], "n": len(pre),
        "sample_ids": [r["sample_id"] for r in pre],
        "prompt_format": "chat (enable_thinking=%s)" % mcfg.get("enable_thinking"),
        "interventions": {"resid_sweep": RESID_SWEEP, "module_scan": MODULE_SCAN,
                          "factorial": FACTORIAL, "steer_alphas": STEER_ALPHAS,
                          "n_null_dirs": N_NULL_DIRS},
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()[:16],
        "date": "2026-10-14",
    }
    json.dump(meta, open(out_dir / "run_metadata.json", "w"), indent=1)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
