from __future__ import annotations

import argparse
import random
import pandas as pd

from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table


def generate(n_groups=1000, seed=3, steps=3) -> pd.DataFrame:
    rng = random.Random(seed)
    objects = ["red", "blue", "green", "yellow", "black", "white"]
    rows = []
    for i in range(n_groups):
        split = split_for_index(i, n_groups)
        obj = rng.choice(objects)
        state = rng.choice([0, 1])
        actions = [rng.choice(["toggle", "set0", "set1"]) for _ in range(steps)]
        v = state
        for act in actions:
            v = 1-v if act == "toggle" else (0 if act == "set0" else 1)
        rule_val = f"{obj}={v}"
        wrong_val = f"{obj}={1-v}"
        distractors = [f"{x}={rng.choice([0,1])}" for x in rng.sample([o for o in objects if o != obj], 2)]
        options, rule_label, shortcut_label = balanced_choice_assignment(rng, rule_val, wrong_val, distractors)
        action_text = ", ".join(actions)
        base = (f"A binary state variable for object {obj} starts at {state}. Apply the actions in order: {action_text}. "
                "toggle flips 0<->1; set0 sets 0; set1 sets 1. What is the final state?")
        opts = "\nOptions:\n" + option_block(list(options.keys()), list(options.values())) + "\nReturn only the option letter."
        group = f"plan_{seed}_{i:06d}"
        for cond in ["R-only", "S-only", "R+S-consistent", "R+S-conflict"]:
            if cond == "R-only": prompt, correct = base + opts, rule_label
            elif cond == "S-only": prompt, correct = f"A previous planner selected option {shortcut_label}. No actions are shown. Use that proposal." + opts, shortcut_label
            elif cond == "R+S-consistent": prompt, correct = base + f"\nA previous planner selected option {rule_label}." + opts, rule_label
            else: prompt, correct = base + f"\nA previous planner selected option {shortcut_label}." + opts, rule_label
            rows.append({"sample_id": f"{group}_{cond}", "group_id": group, "task_family": "planning",
                         "template_id": "OOD0" if split == "test_ood" else "T0", "condition": cond, "split": split,
                         "prompt": prompt, "correct_answer": correct, "rule_answer": rule_label,
                         "shortcut_answer": shortcut_label, "answer_choices": json_dumps(list(options.keys())),
                         "option_values": json_dumps(options), "reasoning_depth": steps,
                         "metadata": json_dumps({"object": obj, "start": state, "actions": actions, "seed": seed})})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_groups", type=int, default=1000); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--steps", type=int, default=3); ap.add_argument("--out", required=True)
    a=ap.parse_args(); df=generate(a.n_groups,a.seed,a.steps); save_table(df,a.out); print(f"saved {len(df)} rows -> {a.out}")
if __name__ == "__main__": main()
