from __future__ import annotations

import argparse
import random

import pandas as pd

from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table


def generate(n_groups=1000, seed=2, steps=3) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n_groups):
        split = split_for_index(i, n_groups)
        start = rng.randint(20, 99)
        ops = []
        value = start
        for _ in range(steps):
            op = rng.choice(["+", "-", "*"])
            if op == "+":
                k = rng.randint(3, 19); value += k
            elif op == "-":
                k = rng.randint(3, min(19, max(3, value - 1))); value -= k
            else:
                k = rng.choice([2, 3]); value *= k
            ops.append((op, k))
        rule_value = str(value)
        shortcut_value = str(value + rng.choice([-3, -2, -1, 1, 2, 3]))
        while shortcut_value == rule_value:
            shortcut_value = str(value + 1)
        distractors = []
        for delta in [5, -5, 7, -7, 9]:
            x = str(value + delta)
            if x not in {rule_value, shortcut_value} and x not in distractors:
                distractors.append(x)
            if len(distractors) == 2:
                break
        options, rule_label, shortcut_label = balanced_choice_assignment(rng, rule_value, shortcut_value, distractors)
        op_text = ", then ".join(f"{op}{k}" for op, k in ops)
        base = f"Start with {start}. Apply these operations in order: {op_text}. What is the final value?"
        opts = "\nOptions:\n" + option_block(list(options.keys()), list(options.values())) + "\nReturn only the option letter."
        group = f"arith_{seed}_{i:06d}"
        for cond in ["R-only", "S-only", "R+S-consistent", "R+S-conflict"]:
            if cond == "R-only":
                prompt = base + opts; correct = rule_label
            elif cond == "S-only":
                prompt = f"A quick calculator reports option {shortcut_label}. No operation list is available. Use the report." + opts
                correct = shortcut_label
            elif cond == "R+S-consistent":
                prompt = base + f"\nA quick calculator reports option {rule_label}." + opts; correct = rule_label
            else:
                prompt = base + f"\nA quick calculator reports option {shortcut_label}." + opts; correct = rule_label
            rows.append({
                "sample_id": f"{group}_{cond}", "group_id": group, "task_family": "arithmetic",
                "template_id": "OOD0" if split == "test_ood" else "T0", "condition": cond, "split": split,
                "prompt": prompt, "correct_answer": correct, "rule_answer": rule_label,
                "shortcut_answer": shortcut_label, "answer_choices": json_dumps(list(options.keys())),
                "option_values": json_dumps(options), "reasoning_depth": steps,
                "metadata": json_dumps({"start": start, "ops": ops, "value": value, "seed": seed}),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_groups", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = generate(a.n_groups, a.seed, a.steps)
    save_table(df, a.out)
    print(f"saved {len(df)} rows -> {a.out}")

if __name__ == "__main__": main()
