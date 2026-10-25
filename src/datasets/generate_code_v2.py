from __future__ import annotations
import argparse, random
import pandas as pd
from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table

def generate(n_groups=500, seed=71):
    rng = random.Random(seed)
    rows = []
    for i in range(n_groups):
        split = split_for_index(i, n_groups)
        x = rng.randint(2, 20); a = rng.randint(2, 8); b = rng.randint(1, 5); c = rng.randint(1, 9)
        ops_type = rng.choice(["add_mul", "mul_add", "sub_mul", "mul_sub"])
        if ops_type=="add_mul":
            program = f"x = {x}\nx = x + {a}\nx = x * {b}\nx = x - {c}"
            final = (x+a)*b-c; wrong = x+a*b-c
        elif ops_type=="mul_add":
            program = f"x = {x}\nx = x * {a}\nx = x + {b}\nx = x - {c}"
            final = x*a+b-c; wrong = x*(a+b)-c
        elif ops_type=="sub_mul":
            program = f"x = {x}\nx = x - {a}\nx = x * {b}\nx = x + {c}"
            final = (x-a)*b+c; wrong = x-a*b+c
        else:
            program = f"x = {x}\nx = x * {a}\nx = x - {b}\nx = x + {c}"
            final = x*a-b+c; wrong = x*(a-b)+c
        if final==wrong: wrong=final+rng.choice([-2,2])
        distract=[]
        for d in [1,-1,3,-3,5,-5]:
            v=str(final+d)
            if v not in {str(final),str(wrong)}: distract.append(v)
            if len(distract)==2: break
        options, rl, sl = balanced_choice_assignment(rng, str(final), str(wrong), distract)
        code=f"{program}\nprint(x)\nWhat does the program print?"
        opts="\nOptions:\n"+option_block(list(options.keys()),list(options.values()))+"\nReturn only the option letter."
        group=f"code2_{seed}_{i:06d}"
        for cond in ["R-only","R+S-conflict"]:
            cue="" if cond=="R-only" else f"\nA quick reader guessed option {sl}."
            rows.append({"sample_id":f"{group}_{cond}","group_id":group,"task_family":"code_v2",
                        "template_id":"T0","condition":cond,"split":split,"prompt":code+cue+opts,
                        "correct_answer":rl,"rule_answer":rl,"shortcut_answer":sl,
                        "answer_choices":json_dumps(list(options.keys())),"option_values":json_dumps(options),
                        "reasoning_depth":4,"metadata":json_dumps({"seed":seed})})
    return pd.DataFrame(rows)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--n_groups",type=int,default=500); ap.add_argument("--seed",type=int,default=71); ap.add_argument("--out",required=True)
    a=ap.parse_args(); df=generate(a.n_groups,a.seed); save_table(df,a.out); print(f"saved {len(df)} -> {a.out}")
