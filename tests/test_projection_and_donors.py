import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.interventions.hooks import make_projection_ablation_hook
from src.metrics.behavior import group_paired_contrast
from src.metrics.generic_health import build_extended_battery


class Identity(nn.Module):
    def forward(self, x):
        return x


def _run(hook):
    m = Identity()
    h = m.register_forward_hook(hook)
    try:
        x = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]).view(2, 1, 3)
        return x, m(x)
    finally:
        h.remove()


def test_projection_removal_math():
    v = torch.tensor([1.0, 0.0, 0.0])
    x, y = _run(make_projection_ablation_hook(v, alpha=1.0, target_pos=0))
    assert torch.allclose(y[0, 0], torch.tensor([0.0, 4.0, 0.0]))
    assert torch.allclose(y[1, 0], torch.tensor([0.0, 0.0, 5.0]))


def test_projection_partial_and_nonunit_vector():
    v = torch.tensor([2.0, 0.0, 0.0])
    x, y = _run(make_projection_ablation_hook(v, alpha=0.5, target_pos=0))
    assert torch.allclose(y[0, 0], torch.tensor([1.5, 4.0, 0.0]))


def test_projection_only_target_position():
    v = torch.tensor([0.0, 0.0, 1.0])
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    m = Identity()
    h = m.register_forward_hook(make_projection_ablation_hook(v, alpha=1.0, target_pos=2))
    try:
        y = m(x)
    finally:
        h.remove()
    assert torch.equal(y[:, 0, :], x[:, 0, :])
    assert torch.equal(y[:, 3, :], x[:, 3, :])
    assert torch.allclose(y[:, 2, 2], torch.zeros(2))


def test_projection_all_positions():
    v = torch.tensor([1.0, 0.0, 0.0])
    x = torch.ones(1, 3, 3)
    m = Identity()
    h = m.register_forward_hook(make_projection_ablation_hook(v, alpha=1.0, all_positions=True))
    try:
        y = m(x)
    finally:
        h.remove()
    assert torch.allclose(y, torch.zeros(1, 3, 3) + torch.tensor([0.0, 1.0, 1.0]))


def test_extended_battery():
    b1 = build_extended_battery()
    assert len(b1) >= 100
    assert b1 == build_extended_battery()
    for prompt, ans in b1:
        assert prompt.strip() and ans.startswith(" ")
    b_small = build_extended_battery(n=60)
    assert len(b_small) == 60


def test_group_paired_contrast_pairs_by_group():
    df = pd.DataFrame({
        "group_id": ["g1", "g1", "g2", "g2", "g3"],
        "condition": ["A", "B", "A", "B", "A"],
        "margin": [2.0, 0.0, 4.0, 1.0, 9.0],
    })
    pc = group_paired_contrast(df, "A", "B")
    assert list(pc.group_id) == ["g1", "g2"]
    assert np.allclose(pc["diff"].values, [2.0, 3.0])


def test_choose_donor():
    from experiments.p1_reasoning_switch.run_patching_controls import choose_donor
    df = pd.DataFrame({
        "group_id": ["g1", "g2", "g3", "g4"],
        "condition": ["R-only"] * 4,
        "rule_answer": ["A", "B", "A", "D"],
        "prompt": ["p1", "p2", "p3", "p4"],
    })
    rng = random.Random(0)
    pool = {}
    for r in df.itertuples():
        pool.setdefault(str(r.rule_answer), []).append(r)
    d = choose_donor(rng, df, "g1", "A", "wrong_donor", pool)
    assert d.group_id != "g1"
    d2 = choose_donor(rng, df, "g3", "A", "same_answer_donor", pool)
    assert d2.group_id != "g3" and str(d2.rule_answer) == "A"
