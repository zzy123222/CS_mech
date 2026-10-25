import math

import numpy as np
import pytest

from src.stats.power import (ci_width_projection, paired_cohens_d_from_groups,
                             required_n_paired, required_n_two_sample)


def test_required_n_paired_classic_value():
    assert required_n_paired(0.5) == 32


def test_required_n_monotone_in_d_and_power():
    assert required_n_paired(0.8) < required_n_paired(0.5)
    assert required_n_paired(0.5, power=0.9) > required_n_paired(0.5, power=0.8)


def test_two_sample_needs_about_double():
    n_p, n_2 = required_n_paired(0.5), required_n_two_sample(0.5)
    assert 2 * n_p - n_2 <= 1 and n_2 >= 2 * n_p - 1


def test_zero_d_raises():
    with pytest.raises(ValueError):
        required_n_paired(0.0)


def test_ci_width_scales_as_one_over_sqrt_n():
    w = ci_width_projection(2.0, [64, 256])
    assert math.isclose(w[256] * 2, w[64], rel_tol=1e-9)


def test_paired_d_from_groups():
    import pandas as pd
    df = pd.DataFrame({
        "group_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
        "condition": ["A", "B"] * 3,
        "margin": [2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
    })
    d, sd, n = paired_cohens_d_from_groups(df, "A", "B")
    assert n == 3 and np.isclose(d, 3.0 / math.sqrt(1.0))
