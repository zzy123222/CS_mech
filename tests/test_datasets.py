import json

import pandas as pd

from src.datasets.generate_graph_reasoning import generate as gen_graph
from src.datasets.generate_arithmetic import generate as gen_arith
from src.datasets.generate_planning import generate as gen_plan
from src.datasets.generate_error_monitor import generate as gen_error, build_check_prompt
from src.datasets.generate_hallucination import generate as gen_hall


def _check_four_condition(df: pd.DataFrame, n_groups: int):
    assert len(df) == 4 * n_groups
    expected = {"R-only", "S-only", "R+S-consistent", "R+S-conflict"}
    assert set(df["condition"]) == expected
    counts = df.groupby("group_id")["condition"].nunique()
    assert (counts == 4).all()
    conflict = df[df.condition == "R+S-conflict"]
    assert (conflict.rule_answer != conflict.shortcut_answer).all()
    assert (conflict.correct_answer == conflict.rule_answer).all()


def test_graph_generator():
    _check_four_condition(gen_graph(20, seed=1), 20)


def test_graph_generator_matched_s():
    df = gen_graph(20, seed=1, extra_matched_s=True)
    assert set(df.condition) == {"R-only", "S-only", "R+S-consistent", "R+S-conflict", "S-matched"}
    sm = df[df.condition == "S-matched"].set_index("group_id")
    conflict = df[df.condition == "R+S-conflict"].set_index("group_id")
    assert set(sm.index) == set(conflict.index)
    for gid, r in sm.iterrows():
        c = conflict.loc[gid]
        assert r.correct_answer == r.shortcut_answer
        assert "Graph facts:" in r.prompt and "A previous quick solver" in r.prompt
        assert r.prompt.count("\n") == c.prompt.count("\n")
        meta = json.loads(r.metadata)
        src = meta["src"]
        for a, _b in meta["facts"]:
            assert a != src


def test_error_monitor_types():
    df = gen_error(24, seed=11, n_steps=4, error_types=["offset", "sign", "carry"])
    assert set(df.error_type) == {"offset", "sign", "carry"}
    for _, r in df.iterrows():
        clean = json.loads(r.clean_values_json)
        err = json.loads(r.error_values_json)
        es = int(r.error_step)
        assert err[es] != clean[es]
        assert err[es - 1] == clean[es - 1]
        if r.error_type == "sign":
            assert err[es] == -clean[es]
        if r.error_type == "carry" and abs(clean[es]) >= 10:
            assert abs(err[es] - clean[es]) == 10


def test_error_monitor_bad_type_rejected():
    import pytest
    with pytest.raises(ValueError):
        gen_error(4, seed=11, error_types=["offset", "typo"])


def test_arithmetic_generator():
    _check_four_condition(gen_arith(20, seed=2), 20)


def test_planning_generator():
    _check_four_condition(gen_plan(20, seed=3), 20)


def test_error_monitor_prefixes():
    df = gen_error(10, seed=11, n_steps=4)
    row = df.iloc[0]
    pre_t = max(1, int(row.error_step) - 1)
    assert build_check_prompt(row, "clean", pre_t) == build_check_prompt(row, "error", pre_t)
    at_t = int(row.error_step)
    assert build_check_prompt(row, "clean", at_t) != build_check_prompt(row, "error", at_t)


def test_hallucination_conditions():
    df = gen_hall(10, seed=71, distractors=5)
    assert len(df) == 40
    assert set(df.condition) == {
        "knowledge_present_clean", "knowledge_absent", "retrieval_failure", "arbitration_failure"
    }
    assert (df.rule_answer != df.shortcut_answer).all()


def test_symbolic_arithmetic_generator():
    import json
    from src.datasets.generate_symbolic_arithmetic import generate as gen_arith2
    df = gen_arith2(15, seed=41)
    assert set(df.condition) == {"R-only", "S-only", "R+S-consistent", "R+S-conflict", "S-matched"}
    assert len(df) == 15 * 5
    conflict = df[df.condition == "R+S-conflict"].set_index("group_id")
    for _, r in df[df.condition == "S-matched"].set_index("group_id").iterrows():
        c = conflict.loc[r.name]
        assert r.prompt.count("\n") == c.prompt.count("\n")
        assert r.correct_answer == r.shortcut_answer
        meta = json.loads(r.metadata)
        assert meta["filler_value"] != meta["value"]
        opts = json.loads(r.option_values)
        assert str(meta["filler_value"]) not in opts.values()
    r0 = df[df.condition == "R-only"].iloc[0]
    m = json.loads(r0.metadata)
    v = m["start"]
    for op, k in m["ops"]:
        v = v + k if op == "+" else (v - k if op == "-" else v * k)
    opts = json.loads(r0.option_values)
    assert opts[r0.rule_answer] == str(v)


def test_string_program_generator():
    import json
    from src.datasets.generate_string_program import generate as gen_str
    df = gen_str(10, seed=51)
    assert set(df.condition) == {"R-only", "S-only", "R+S-consistent", "R+S-conflict", "S-matched"}
    conflict = df[df.condition == "R+S-conflict"].set_index("group_id")
    for _, r in df[df.condition == "S-matched"].set_index("group_id").iterrows():
        c = conflict.loc[r.name]
        assert r.prompt.count("\n") == c.prompt.count("\n")
        assert r.correct_answer == r.shortcut_answer
        assert "Operations in order" in r.prompt
        assert r.prompt.count("\n") == c.prompt.count("\n")
    r0 = df[df.condition == "R-only"].iloc[0]
    m = json.loads(r0.metadata)
    opts = json.loads(r0.option_values)
    assert opts[r0.rule_answer] == " ".join(m["rule"])
