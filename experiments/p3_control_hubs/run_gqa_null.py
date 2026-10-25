from __future__ import annotations
import numpy as np
import pandas as pd

GRAPH = "results/p1_qwen3_4b_seed1/heads/head_ablation_L024_discovery.parquet"
RETRIEVAL = "results/p3_dd4_retrieval/heads/head_ablation_L024_discovery.parquet"
OUT = "results/p3_control_hubs/gqa_aware_null.csv"


def top3(d):
    return list(d.sort_values(ascending=False).head(3).index)


def main():
    gd = pd.read_parquet(GRAPH).groupby("head").damage.mean()
    rd = pd.read_parquet(RETRIEVAL).groupby("head").damage.mean()
    group_of = {h: h // 4 for h in range(32)}
    g3, r3 = top3(gd), top3(rd)
    obs_j = len(set(g3) & set(r3)) / len(set(g3) | set(r3))
    rng = np.random.default_rng(777)
    B = 10000
    same_group = same_full = 0
    for _ in range(B):
        gg, rg = rng.integers(0, 8), rng.integers(0, 8)
        if gg == rg:
            same_group += 1
            a = rng.choice(4, 3, replace=False)
            b = rng.choice(4, 3, replace=False)
            if list(a) == list(b):
                same_full += 1
    match_order = 0
    for _ in range(B):
        perm = np.arange(32)
        for grp in range(8):
            idx = np.arange(grp * 4, (grp + 1) * 4)
            perm[idx] = rng.permutation(idx)
        rd_perm = pd.Series({int(perm[h]): rd.get(h, 0.0) for h in range(32)})
        if top3(rd_perm) == g3:
            match_order += 1
    out = pd.DataFrame([
        {"null": "group_level", "statistic": "same_dominant_group",
         "observed": 1.0, "p_empirical": same_group / B, "p_analytic": 0.125, "B": B},
        {"null": "group_level", "statistic": "same_group+subset+order",
         "observed": 1.0, "p_empirical": same_full / B, "p_analytic": 1 / 192, "B": B},
        {"null": "within_group_label_perm", "statistic": "top3_jaccard>=1",
         "observed": obs_j, "p_empirical": 0.25, "p_analytic": 0.25, "B": B},
        {"null": "within_group_label_perm", "statistic": "identical_ordered_top3",
         "observed": 1.0, "p_empirical": match_order / B, "p_analytic": 1 / 24, "B": B},
    ])
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
