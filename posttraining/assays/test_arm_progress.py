from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "posttraining/results/formal"
OUT = FORMAL / "analysis"

ARMS = [("general_sft_seed1", "A", 1), ("correction_sft_seed1", "B", 1), ("sham_correction_seed1", "C", 1),
        ("general_sft_seed2", "A", 2), ("correction_sft_seed2", "B", 2), ("sham_correction_seed2", "C", 2),
        ("general_sft_seed3", "A", 3), ("correction_sft_seed3", "B", 3), ("sham_correction_seed3", "C", 3)]


def ckpt_samples(npz, step, tag):
    if tag == "integrator":
        return np.concatenate([npz[f"step{step:04d}_integrator_graph_per_sample"],
                               npz[f"step{step:04d}_integrator_retrieval_per_sample"]])
    return npz[f"step{step:04d}_{tag}_per_sample"]


def build_long(tag):
    ref_npz = np.load(ROOT / "posttraining/results/reference_metrics.npz")
    if tag == "integrator":
        ref_s = np.concatenate([ref_npz["integrator_graph_per_sample"],
                                ref_npz["integrator_retrieval_per_sample"]])
    else:
        ref_s = ref_npz[f"{tag}_per_sample"]
    ref_mean = float(ref_s.mean())

    rows = []
    for arm_dir, arm, seed in ARMS:
        d = FORMAL / arm_dir
        if not (d / "mechanism_panel.csv").exists():
            continue
        npz = np.load(d / "mechanism_panel_persample.npz")
        steps = sorted({int(k.split("_")[0][4:]) for k in npz.files if k.startswith("step")})
        base_s = ckpt_samples(npz, steps[0], tag)
        base_mean = float(base_s.mean())
        denom = ref_mean - base_mean
        if abs(denom) < 1e-9:
            return None, None
        for step in steps:
            s = ckpt_samples(npz, step, tag)
            for i, v in enumerate(s):
                rows.append({"sample": f"{arm_dir}:{i}", "arm": arm, "seed": seed,
                             "step": step, "x": np.log1p(step), "D": (v - base_mean) / denom})
    return pd.DataFrame(rows), ref_mean


def fit_ols(df):
    X = np.column_stack([
        np.ones(len(df)),
        (df.arm == "B").values.astype(float),
        (df.arm == "C").values.astype(float),
        df.x.values,
        (df.arm == "B").values * df.x.values,
        (df.arm == "C").values * df.x.values,
    ])
    y = df.D.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"intercept": beta[0], "arm_B": beta[1], "arm_C": beta[2],
            "x": beta[3], "B:x": beta[4], "C:x": beta[5]}


def cluster_bootstrap(df, n_boot=2000, seed=123):
    rng = np.random.default_rng(seed)
    clusters = df["sample"].unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [df[df["sample"] == c] for c in pick]
        bdf = pd.concat(parts)
        try:
            boots.append(fit_ols(bdf))
        except Exception:
            continue
    return pd.DataFrame(boots)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, tag, name in [("controller_d_L22mlp", "controller", "Controller (L22:mlp)"),
                            ("integrator_d_L24attn", "integrator", "Integrator (L24:attn)")]:
        df, ref_mean = build_long(tag)
        if df is None or df.seed.nunique() < 1:
            print(f"[skip] {name}: unstable reference denominator")
            continue
        betas = fit_ols(df)
        boots = cluster_bootstrap(df)
        out = {"n_obs": int(len(df)), "n_clusters": int(df["sample"].nunique()),
               "seeds": sorted(df.seed.unique().tolist()),
               "coefficients": {k: round(v, 4) for k, v in betas.items()}}
        for c in ["B:x", "C:x", "arm_B"]:
            lo, hi = np.percentile(boots[c], [2.5, 97.5])
            p = 2 * min((boots[c] <= 0).mean(), (boots[c] >= 0).mean())
            out[f"{c}_ci"] = [round(lo, 4), round(hi, 4)]
            out[f"{c}_p"] = round(float(p), 4)
        results[name] = out
        print(f"\n=== {name} ===")
        print(json.dumps(out, indent=2))

    dfc, _ = build_long("controller")
    if dfc is not None:
        pc = dfc.groupby(["seed", "arm", "step"]).D.mean().reset_index()
        piv = pc.pivot_table(index=["seed", "step"], columns="arm", values="D")
        piv["B_minus_C"] = piv["B"] - piv["C"]
        piv["B_minus_A"] = piv["B"] - piv["A"]
        piv.round(4).to_csv(OUT / "arm_contrast_by_checkpoint.csv")
        print("\nper-checkpoint contrasts saved")

    with open(OUT / "arm_progress_model.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {OUT / 'arm_progress_model.json'}")


if __name__ == "__main__":
    main()
