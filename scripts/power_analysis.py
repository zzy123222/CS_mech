from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.stats.power import (ci_width_projection, paired_cohens_d_from_groups,
                             required_n_paired, required_n_two_sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior_parquet", default=None,
                    help="Optional behavior.parquet with condition/margin/group_id columns")
    ap.add_argument("--cond_a", default="R+S-conflict")
    ap.add_argument("--cond_b", default="S-only")
    ap.add_argument("--effect_d", type=float, default=None,
                    help="Direct effect size if no parquet given")
    ap.add_argument("--powers", nargs="*", type=float, default=[0.8, 0.9])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--candidate_n", nargs="*", type=int, default=[64, 128, 256, 512, 1024])
    args = ap.parse_args()

    sd_diff = None
    if args.behavior_parquet:
        import pandas as pd
        df = pd.read_parquet(args.behavior_parquet)
        d, sd_diff, n = paired_cohens_d_from_groups(df, args.cond_a, args.cond_b)
        print(f"pilot: {args.cond_a} vs {args.cond_b}  paired d={d:.3f}  sd_diff={sd_diff:.3f}  n_pairs={n}")
    elif args.effect_d is not None:
        d = args.effect_d
        print(f"using supplied d={d:.3f}")
    else:
        ap.error("provide --behavior_parquet or --effect_d")

    print(f"\nrequired N (paired, two-sided, alpha={args.alpha}):")
    for p in args.powers:
        print(f"  power={p:.2f}: N={required_n_paired(d, power=p, alpha=args.alpha)}")
    print(f"required N per group (two-sample):")
    for p in args.powers:
        print(f"  power={p:.2f}: N={required_n_two_sample(d, power=p, alpha=args.alpha)}")

    if sd_diff:
        widths = ci_width_projection(sd_diff, args.candidate_n, alpha=args.alpha)
        print(f"\nprojected paired-mean 95% CI widths (sd_diff={sd_diff:.3f}):")
        for n, w in widths.items():
            print(f"  N={n:5d}: +/-{w/2:.3f}  (width {w:.3f})")


if __name__ == "__main__":
    main()
