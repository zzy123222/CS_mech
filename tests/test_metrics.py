import numpy as np
import pandas as pd

from src.metrics.behavior import (
    cohens_d, normalized_recovery, rescue_rate, reasoning_reliance_index,
    pleiotropy_index,
)
from src.stats.bootstrap import paired_bootstrap
from src.stats.fdr import bh_fdr


def test_rri():
    df = pd.DataFrame({
        "condition": ["S-only", "S-only", "R+S-conflict", "R+S-conflict"],
        "margin": [-2.0, -1.0, 1.0, 2.0],
    })
    assert np.isclose(reasoning_reliance_index(df), 3.0)


def test_recovery_and_rescue():
    assert np.isclose(normalized_recovery(10, 0, 5), 0.5)
    assert np.isclose(rescue_rate(10, 2, 6), 0.5)


def test_pleiotropy_index_range():
    assert np.isclose(pleiotropy_index([1, 1, 1, 1]), 1.0)
    assert pleiotropy_index([1, 0, 0, 0]) < 0.3


def test_bootstrap_and_fdr():
    out = paired_bootstrap(np.arange(10.0) + 1, np.arange(10.0), n_boot=200, seed=0)
    assert out["mean_diff"] > 0.9
    q, reject = bh_fdr([0.001, 0.02, 0.8], alpha=0.05)
    assert len(q) == 3 and len(reject) == 3
    assert reject[0]


def test_cohens_d_finite():
    assert np.isfinite(cohens_d([1, 2, 3], [0, 1, 2]))
