from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "posttraining/results/formal"
OUT = FORMAL / "analysis"

BASE_V = -0.0267152190208435
REF_V = -0.33984358867701303
ARMS = [("general_sft", "A"), ("correction_sft", "B"), ("sham_correction", "C")]


def load_run(arm, seed):
    d = FORMAL / f"{arm}_seed{seed}"
    npz = np.load(d / "mechanism_panel_persample.npz")
    steps = sorted({int(k.split("_")[0][4:]) for k in npz.files if k.startswith("step")})
    data = []
    for step in steps:
        arr = npz[f"step{step:04d}_controller_per_sample"]
        for i, v in enumerate(arr):
            data.append((step, i, (v - BASE_V) / (REF_V - BASE_V)))
    return pd.DataFrame(data, columns=["step", "sample", "D"])


def slope(df):
    x = np.log1p(df.step.values)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, df.D.values, rcond=None)
    return float(beta[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, slopes = {}, {}
    for arm, key in ARMS:
        for seed in (1, 2, 3):
            df = load_run(arm, seed)
            runs[(key, seed)] = df
            slopes[(key, seed)] = slope(df)

    slope_rows = [{"arm": k, "seed": s, "slope_D_per_logstep": round(v, 4)}
                  for (k, s), v in slopes.items()]
    print("=== run-level controller slopes (D ~ log(1+step)) ===")
    for r in slope_rows:
        print(f"  {r['arm']} seed{r['seed']}: {r['slope_D_per_logstep']:+.4f}")

    def med(vals): return float(np.median(vals))
    summary = {
        "slopes_A": [round(slopes[("A", s)], 4) for s in (1, 2, 3)],
        "slopes_B": [round(slopes[("B", s)], 4) for s in (1, 2, 3)],
        "slopes_C": [round(slopes[("C", s)], 4) for s in (1, 2, 3)],
        "median_slope_A": round(med([slopes[("A", s)] for s in (1, 2, 3)]), 4),
        "median_slope_B": round(med([slopes[("B", s)] for s in (1, 2, 3)]), 4),
        "median_slope_C": round(med([slopes[("C", s)] for s in (1, 2, 3)]), 4),
        "B_minus_A_positive_seeds": int(sum(slopes[("B", s)] > slopes[("A", s)] for s in (1, 2, 3))),
        "B_minus_C_positive_seeds": int(sum(slopes[("B", s)] > slopes[("C", s)] for s in (1, 2, 3))),
    }
    print("\n=== seed-stratified summary ===")
    print(json.dumps(summary, indent=2))

    rng = np.random.default_rng(2026)
    n_boot = 2000
    seed_list = [1, 2, 3]
    run_arrays = {}
    for arm, key in ARMS:
        for s in seed_list:
            df = runs[(key, s)]
            run_arrays[(key, s)] = (np.log1p(df.step.values).astype(np.float64),
                                    df.D.values.astype(np.float64))
    n_rows_total = sum(len(v[1]) for v in run_arrays.values())
    boots = np.empty((n_boot, 3))
    for b in range(n_boot):
        xs, ds, bs, cs = [], [], [], []
        for arm, key in ARMS:
            picked = rng.choice(seed_list, size=3, replace=True)
            for s in picked:
                x, D = run_arrays[(key, s)]
                idx = rng.integers(0, len(D), len(D))
                xs.append(x[idx]); ds.append(D[idx])
                bs.append(np.full(len(idx), 1.0 if key == "B" else 0.0))
                cs.append(np.full(len(idx), 1.0 if key == "C" else 0.0))
        xv = np.concatenate(xs); dv = np.concatenate(ds)
        bv = np.concatenate(bs); cv = np.concatenate(cs)
        X = np.column_stack([np.ones(len(xv)), bv, cv, xv, bv * xv, cv * xv])
        beta, *_ = np.linalg.lstsq(X, dv, rcond=None)
        boots[b] = [beta[4], beta[5], beta[4] - beta[5]]
    hier = {}
    for j, c in enumerate(["B:x", "C:x", "B:x_minus_C:x"]):
        col = boots[:, j]
        lo, hi = np.percentile(col, [2.5, 97.5])
        p = 2 * min((col <= 0).mean(), (col >= 0).mean())
        hier[c] = {"median": round(float(np.median(col)), 4),
                   "ci95": [round(float(lo), 4), round(float(hi), 4)],
                   "p_two_sided": round(float(p), 4)}
    print("\n=== hierarchical bootstrap (sensitivity; run-first resampling) ===")
    print(json.dumps(hier, indent=2))

    out = {"run_level_slopes": slope_rows,
           "seed_stratified_summary": summary,
           "hierarchical_bootstrap": hier,
           "note": ("Sample-cluster CIs from the main model treat evaluation samples as the "
                    "resampling unit; hierarchical CIs resample the 3 dataset-generation runs "
                    "first and are reported as sensitivity analysis only.")}
    with open(OUT / "run_level_robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(slope_rows).to_csv(OUT / "run_level_slopes.csv", index=False)
    print(f"\nsaved: {OUT / 'run_level_robustness.json'}")


if __name__ == "__main__":
    main()
