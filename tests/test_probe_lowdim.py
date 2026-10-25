import numpy as np

from src.probes.linear_probe import centroid_direction, fit_binary_probe, projection_auc


def _synthetic(n=600, d=40, shift=2.5, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros(d); v[0] = 1.0
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, size=n)
    X[y == 1] += shift * v
    return X, y


def test_projection_auc_detects_single_direction_signal():
    X, y = _synthetic()
    tr, te = slice(0, 450), slice(450, None)
    v = centroid_direction(X[tr][y[tr] == 1], X[tr][y[tr] == 0])
    auc_1d = projection_auc(X[te], y[te], v)
    full = fit_binary_probe(X[tr], y[tr], X[te], y[te])
    assert auc_1d > 0.85
    assert full["auc"] > 0.9
    assert full["auc"] - auc_1d < 0.05


def test_projection_auc_is_direction_sensitive():
    X, y = _synthetic()
    v = centroid_direction(X[:450][y[:450] == 1], X[:450][y[:450] == 0])
    assert projection_auc(X[450:], y[450:], v) > 0.85
    assert projection_auc(X[450:], y[450:], -v) < 0.15
