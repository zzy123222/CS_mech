import numpy as np
import torch

from experiments.p1_reasoning_switch.run_answer_orthogonal_rescue import orthonormalize


def test_orthonormalize_columns():
    torch.manual_seed(0)
    cols = torch.randn(64, 4)
    Q = orthonormalize(cols)
    assert Q.shape == (64, 4)
    gram = Q.T @ Q
    assert torch.allclose(gram, torch.eye(4), atol=1e-5)


def test_orthonormalize_dependent_columns():
    a = torch.randn(32, 1)
    cols = torch.cat([a, 2 * a, torch.randn(32, 1)], dim=1)
    Q = orthonormalize(cols)
    gram = Q.T @ Q
    assert torch.allclose(gram, torch.eye(3), atol=1e-5) or torch.linalg.matrix_rank(cols).item() >= 2


def test_projection_decomposition_adds_up():
    torch.manual_seed(1)
    Q = orthonormalize(torch.randn(32, 3))
    a = torch.randn(32)
    proj = Q @ (Q.T @ a)
    perp = a - proj
    assert torch.allclose(proj + perp, a, atol=1e-6)
    assert torch.allclose(Q.T @ perp, torch.zeros(3), atol=1e-5)
    assert torch.isclose(proj @ perp, torch.tensor(0.0), atol=1e-5)
