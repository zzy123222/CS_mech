from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table

TRAIN_TEMPLATES = [
    "Execute the program step by step.",
    "Apply each instruction to the string in order.",
    "Run the operations on the initial string.",
]
OOD_TEMPLATES = [
    "Carry the string through the program below.",
    "Process the initial string with the given operations.",
]

SYMBOLS = list("GHJKLMNP")
OPS = ["rotl", "rev", "swap01", "swap02", "swap13", "repl"]


def _apply(s, op, rng):
    if op == "rotl":
        return s[1:] + s[0]
    if op == "rev":
        return s[::-1]
    if op == "swap01":
        return s[1] + s[0] + s[2:]
    if op == "swap02":
        return s[2] + s[1] + s[0] + s[3]
    if op == "swap13":
        return s[0] + s[3] + s[2] + s[1]
    if op == "repl":
        x, y = rng.sample(list(set(s)), 2)
        return s.replace(x, y, 1)
    raise ValueError(op)


def _op_text(op, rng, s):
    if op == "rotl":
        return "rotate-left"
    if op == "rev":
        return "reverse"
    if op in ("swap01", "swap02", "swap13"):
        return f"swap({op[-2]},{op[-1]})"
    if op == "repl":
        return "replace-first"
    raise ValueError(op)


def _noop_ops(rng):
    return ["swap(0,0)", "replace-first-with-itself", "rotate x4"]


def generate(n_groups=1000, seed=51, n_ops=3) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n_groups):
        split = split_for_index(i, n_groups)
        bank = OOD_TEMPLATES if split == "test_ood" else TRAIN_TEMPLATES
        tid = rng.randrange(len(bank))
        template_id = f"{'OOD' if split == 'test_ood' else 'T'}{tid:02d}"
        s0 = "".join(rng.sample(SYMBOLS, 4))
        ops = [rng.choice(OPS) for _ in range(n_ops)]
        cur = s0
        states = [s0]
        for op in ops:
            cur = _apply(cur, op, rng)
            states.append(cur)
        rule = states[-1]
        short_ops = [o for j, o in enumerate(ops) if j != 1]
        scur = s0
        for op in short_ops:
            scur = _apply(scur, op, rng)
        shortcut = scur
        while shortcut == rule:
            scur = _apply(scur, rng.choice(OPS), rng)
            shortcut = scur
        distract = []
        for cand in [states[2], states[1], states[-2]]:
            if cand not in {rule, shortcut} and cand not in distract:
                distract.append(cand)
            if len(distract) == 2:
                break
        while len(distract) < 2:
            c = list(rule); rng.shuffle(c); c = "".join(c)
            if c not in {rule, shortcut} and c not in distract:
                distract.append(c)
        fmt = lambda x: " ".join(x)
        options, rule_label, shortcut_label = balanced_choice_assignment(
            rng, fmt(rule), fmt(shortcut), [fmt(d) for d in distract])
        opnames = ", ".join(_op_text(o, rng, s0) for o in ops)
        noopnames = ", ".join(_noop_ops(rng))
        question = f"{bank[tid]} Initial string: {fmt(s0)}. Operations in order: {opnames}. What is the final string?"
        q_mask = f"{bank[tid]} Initial string: {fmt(s0)}. The operations are hidden. What is the final string?"
        q_noop = f"{bank[tid]} Initial string: {fmt(s0)}. Operations in order: {noopnames}. What is the final string?"
        cue = "A quick solver reports option {L}."
        opts = "\nOptions:\n" + option_block(list(options.keys()), list(options.values())) + "\nReturn only the option letter."
        group = f"strprog_{seed}_{i:06d}"
        cond_prompts = {
            "R-only": (question, None, rule_label),
            "S-only": (q_mask, cue.format(L=shortcut_label) + " Use the report.", shortcut_label),
            "R+S-consistent": (question, cue.format(L=rule_label), rule_label),
            "R+S-conflict": (question, cue.format(L=shortcut_label), rule_label),
            "S-matched": (q_noop, cue.format(L=shortcut_label), shortcut_label),
        }
        for cond, (q, cue_text, correct) in cond_prompts.items():
            lines = [q] + ([cue_text] if cue_text else [])
            rows.append({
                "sample_id": f"{group}_{cond}", "group_id": group,
                "task_family": "string_program", "template_id": template_id,
                "condition": cond, "split": split,
                "prompt": "\n".join(lines) + opts,
                "correct_answer": correct, "rule_answer": rule_label,
                "shortcut_answer": shortcut_label,
                "answer_choices": json_dumps(list(options.keys())),
                "option_values": json_dumps(options),
                "reasoning_depth": n_ops,
                "metadata": json.dumps({"seed": seed, "s0": s0, "ops": ops,
                                         "rule": rule, "shortcut": shortcut}),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_groups", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--n_ops", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = generate(a.n_groups, a.seed, a.n_ops)
    save_table(df, a.out)
    print(f"saved {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
