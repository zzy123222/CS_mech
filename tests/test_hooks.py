import torch
import torch.nn as nn

from src.interventions.hooks import (
    make_patch_hook, make_mean_ablation_hook, make_zero_ablation_hook, make_steer_hook
)


class Identity(nn.Module):
    def forward(self, x):
        return x


def _run(hook):
    m = Identity()
    h = m.register_forward_hook(hook)
    try:
        x = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
        return x, m(x)
    finally:
        h.remove()


def test_patch_only_target_position():
    source = torch.tensor([100.0, 101.0, 102.0])
    x, y = _run(make_patch_hook(source, target_pos=2))
    assert torch.equal(y[:, 2, :], source.expand(2, -1))
    assert torch.equal(y[:, 0, :], x[:, 0, :])


def test_mean_ablation_negative_index():
    mean = torch.tensor([9.0, 9.0, 9.0])
    x, y = _run(make_mean_ablation_hook(mean, target_pos=-1))
    assert torch.equal(y[:, -1, :], mean.expand(2, -1))
    assert torch.equal(y[:, 0, :], x[:, 0, :])


def test_steering_adds_vector():
    v = torch.tensor([1.0, -2.0, 3.0])
    x, y = _run(make_steer_hook(v, alpha=0.5, target_pos=1))
    assert torch.allclose(y[:, 1, :], x[:, 1, :] + 0.5 * v)


def test_zero_ablation():
    _, y = _run(make_zero_ablation_hook(target_pos=0))
    assert torch.equal(y[:, 0, :], torch.zeros_like(y[:, 0, :]))
