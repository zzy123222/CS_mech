from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import pandas as pd

from src.metrics.behavior import group_paired_contrast
from src.stats.bootstrap import paired_bootstrap

DEFAULT_CONTRASTS = [
    ("R+S-conflict", "S-only"),
    ("R+S-conflict", "R+S-consistent"),
    ("R-only", "S-matched"),
    ("R-only", "S-only"),
    ("R+S-consistent", "R+S-conflict"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default="results/p1_qwen3_4b_seed1/behavior.parquet")
    ap.add_argument("--contrasts", nargs="*", default=None,
                    help="a:b pairs; default set covers the report's key contrasts")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = pd.read_parquet(args.behavior)
    contrasts = []
    for c in (args.contrasts or []):
        a, b = c.split(":", 1)
        contrasts.append((a, b))
    if not contrasts:
        contrasts = DEFAULT_CONTRASTS
    rows = []
    for a, b in contrasts:
        pc = group_paired_contrast(df, a, b)
        if len(pc) < 5:
            continue
        bs = paired_bootstrap(pc.a.values, pc.b.values, n_boot=args.n_boot, seed=args.seed)
        dz = float(pc["diff"].mean() / pc["diff"].std(ddof=1)) if pc["diff"].std(ddof=1) > 0 else float("nan")
        rows.append({"contrast": f"{a} - {b}", "n_groups": len(pc),
                     "mean_diff": bs["mean_diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                     "p_two_sided": bs["p_two_sided"], "cohens_dz": dz})
    out = pd.DataFrame(rows)
    out_path = Path(args.out or (Path(args.behavior).parent / "behavior_paired_stats.csv"))
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
