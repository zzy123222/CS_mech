from __future__ import annotations
import argparse, random
import pandas as pd
from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table

def generate(n_groups=500, seed=61):
    rng = random.Random(seed)
    objects = ["red", "blue", "green", "yellow", "black", "white"]
    rows = []
    for i in range(n_groups):
        split = split_for_index(i, n_groups)
        obj = rng.choice(objects)
        state = rng.choice([0, 1])
        steps = rng.randint(3, 5)
        actions = [rng.choice(["toggle", "set0", "set1"]) for _ in range(steps)]
        v = state
        for act in actions:
            v = 1-v if act=="toggle" else (0 if act=="set0" else 1)
        rule_val = f"{obj}={v}"
        wrong_val = f"{obj}={1-v}"
        distract = [f"{x}={rng.choice([0,1])}" for x in rng.sample([o for o in objects if o!=obj], 2)]
        options, rl, sl = balanced_choice_assignment(rng, rule_val, wrong_val, distract)
        action_text = ", ".join(actions)
        base = f"A binary state variable for {obj} starts at {state}. Apply: {action_text}. Final state?"
        opts = "\nOptions:\n" + option_block(list(options.keys()), list(options.values())) + "\nReturn only the option letter."
        group = f"plan2_{seed}_{i:06d}"
        for cond in ["R-only","S-only","R+S-consistent","R+S-conflict"]:
            if cond=="R-only": prompt, correct = base+opts, rl
            elif cond=="S-only": prompt = f"A planner selected option {sl}. No state info. Use it."+opts; correct = sl
            elif cond=="R+S-consistent": prompt = base+f"\nA planner selected option {rl}."+opts; correct = rl
            else: prompt = base+f"\nA planner selected option {sl}."+opts; correct = rl
            rows.append({"sample_id":f"{group}_{cond}","group_id":group,"task_family":"planning_v2",
                        "template_id":"T0","condition":cond,"split":split,"prompt":prompt,
                        "correct_answer":correct,"rule_answer":rl,"shortcut_answer":sl,
                        "answer_choices":json_dumps(list(options.keys())),"option_values":json_dumps(options),
                        "reasoning_depth":steps,"metadata":json_dumps({"seed":seed})})
    return pd.DataFrame(rows)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--n_groups",type=int,default=500); ap.add_argument("--seed",type=int,default=61); ap.add_argument("--out",required=True)
    a=ap.parse_args(); df=generate(a.n_groups,a.seed); save_table(df,a.out); print(f"saved {len(df)} -> {a.out}")
