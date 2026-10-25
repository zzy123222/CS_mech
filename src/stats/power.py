from __future__ import annotations

import math

from scipy import stats


def required_n_paired(d: float, power: float = 0.8, alpha: float = 0.05) -> int:
    if abs(d) < 1e-9:
        raise ValueError("effect size d must be non-zero")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(math.ceil(((z_a + z_b) / abs(d)) ** 2))


def required_n_two_sample(d: float, power: float = 0.8, alpha: float = 0.05) -> int:
    if abs(d) < 1e-9:
        raise ValueError("effect size d must be non-zero")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(math.ceil(2 * ((z_a + z_b) / abs(d)) ** 2))


def paired_cohens_d_from_groups(df, cond_a: str, cond_b: str, key: str = "group_id",
                                value: str = "margin"):
    a = df[df.condition == cond_a].set_index(key)[value]
    b = df[df.condition == cond_b].set_index(key)[value]
    ids = a.index.intersection(b.index)
    diff = (a.loc[ids] - b.loc[ids]).astype(float)
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan"), n
    sd = float(diff.std(ddof=1))
    d = float(diff.mean() / sd) if sd > 0 else float("nan")
    return d, sd, n


def ci_width_projection(sd_diff: float, n_values, alpha: float = 0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    return {int(n): float(2 * z * sd_diff / math.sqrt(n)) for n in n_values}
