from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table

TRAIN_TEMPLATES = [
    "Compute the final value step by step.",
    "Evaluate the running result after every operation.",
    "Work through the operations in order and find the result.",
]
OOD_TEMPLATES = [
    "Track the running total through the operations below.",
    "Carry the value through each step and report the end result.",
]

CONDITIONS = ["R-only", "S-only", "R+S-consistent", "R+S-conflict"]


def _chain(start, ops):
    vals, v = [start], start
    for op, k in ops:
        v = v + k if op == "+" else (v - k if op == "-" else v * k)
        vals.append(v)
    return vals


def _ops_text(ops):
    sym = {"+": "+", "-": "-", "*": "x"}
    return ", ".join(f"{sym[op]}{k}" for op, k in ops)


def _steps_block(vals, ops):
    lines = ["Worked steps:"]
    for i, (op, k) in enumerate(ops):
        lines.append(f"Step {i+1}: {vals[i]} {op if op != '*' else 'x'} {k} = {vals[i+1]}")
    return lines


def _make(rng, steps, hard=False):
    start = rng.randint(20, 99)
    ops = []
    for _ in range(steps):
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            k = rng.randint(23, 49) if hard else rng.randint(6, 19)
        elif op == "-":
            k = rng.randint(3, min(12, max(3, 400 - start)))
            if hard:
                k = rng.randint(13, 47)
        else:
            k = rng.choice([6, 7, 8, 9, 11, 12]) if hard else rng.choice([2, 3])
        ops.append((op, k))
    return start, ops


def generate(n_groups=1000, seed=41, steps=3, hard=False, mode="verify") -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n_groups):
        split = split_for_index(i, n_groups)
        bank = OOD_TEMPLATES if split == "test_ood" else TRAIN_TEMPLATES
        tid = rng.randrange(len(bank))
        template = bank[tid]
        template_id = f"{'OOD' if split == 'test_ood' else 'T'}{tid:02d}"
        start, ops = _make(rng, steps, hard=hard)
        vals = _chain(start, ops)
        rule_value = vals[-1]
        shortcut_value = rule_value + rng.choice([-3, -2, -1, 1, 2, 3])
        f_start = start + rng.choice([17, 23, 31])
        f_vals = _chain(f_start, ops)
        while f_vals[-1] in (rule_value, shortcut_value):
            f_start += 7
            f_vals = _chain(f_start, ops)
        def _perturb(shift):
            return [(op, k + shift + 2 * j) for j, (op, k) in enumerate(ops)]
        f_ops = _perturb(3)
        f_rule = _chain(start, f_ops)[-1]
        tries = 0
        while f_rule in (rule_value, shortcut_value) and tries < 100:
            tries += 1
            f_ops = _perturb(3 + 7 * tries)
            f_rule = _chain(start, f_ops)[-1]
        distract = []
        for delta in [5, -5, 7, -7, 9, -9]:
            x = rule_value + delta
            if x not in {rule_value, shortcut_value} and x != f_vals[-1] and x != f_rule:
                distract.append(str(x))
            if len(distract) == 2:
                break
        options, rule_label, shortcut_label = balanced_choice_assignment(
            rng, str(rule_value), str(shortcut_value), distract)
        cue = "A quick solver reports option {L}."
        opts = "\nOptions:\n" + option_block(list(options.keys()), list(options.values())) + "\nReturn only the option letter."
        group = f"arith_{seed}_{i:06d}"
        if mode == "verify":
            question = f"{template} Start with {start}. Apply in order: {_ops_text(ops)}. What is the final value?"
            steps_lines = _steps_block(vals, ops)
            filler_lines = _steps_block(f_vals, ops)
            cond_prompts = {
                "R-only": ([question] + steps_lines, None, rule_label),
                "S-only": ([question], cue.format(L=shortcut_label) + " No worked steps are available. Use the report.", shortcut_label),
                "R+S-consistent": ([question] + steps_lines, cue.format(L=rule_label), rule_label),
                "R+S-conflict": ([question] + steps_lines, cue.format(L=shortcut_label), rule_label),
                "S-matched": ([question] + filler_lines, cue.format(L=shortcut_label), shortcut_label),
            }
        else:
            q_real = f"{template} Start with {start}. Apply in order: {_ops_text(ops)}. What is the final value?"
            q_mask = f"{template} Start with {start}. Apply in order: {', '.join('?' * len(str(k)) for _, k in ops)}. What is the final value?"
            q_filler = f"{template} Start with {start}. Apply in order: {_ops_text(f_ops)}. What is the final value?"
            cond_prompts = {
                "R-only": ([q_real], None, rule_label),
                "S-only": ([q_mask], cue.format(L=shortcut_label) + " The operands are hidden. Use the report.", shortcut_label),
                "R+S-consistent": ([q_real], cue.format(L=rule_label), rule_label),
                "R+S-conflict": ([q_real], cue.format(L=shortcut_label), rule_label),
                "S-matched": ([q_filler], cue.format(L=shortcut_label), shortcut_label),
            }
        for cond, (lead, cue_text, correct) in cond_prompts.items():
            lines = lead + ([cue_text] if cue_text else [])
            rows.append({
                "sample_id": f"{group}_{cond}", "group_id": group,
                "task_family": "symbolic_arithmetic", "template_id": template_id,
                "condition": cond, "split": split,
                "prompt": "\n".join(lines) + opts,
                "correct_answer": correct, "rule_answer": rule_label,
                "shortcut_answer": shortcut_label,
                "answer_choices": json_dumps(list(options.keys())),
                "option_values": json_dumps(options),
                "reasoning_depth": steps,
                "metadata": json_dumps({"seed": seed, "start": start, "ops": ops,
                                         "value": rule_value, "filler_value": f_vals[-1],
                                         "mode": mode}),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_groups", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--hard", action="store_true", help="larger operands: more computation pressure")
    ap.add_argument("--mode", choices=["verify", "compute"], default="verify")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = generate(a.n_groups, a.seed, a.steps, hard=a.hard, mode=a.mode)
    save_table(df, a.out)
    print(f"saved {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
