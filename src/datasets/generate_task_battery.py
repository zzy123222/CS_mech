from __future__ import annotations

import argparse
import json
import random
import pandas as pd

from src.datasets.generate_graph_reasoning import generate as gen_graph
from src.datasets.generate_arithmetic import generate as gen_arith
from src.datasets.generate_planning import generate as gen_plan
from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table


def gen_code(n_groups=500, seed=41):
    rng=random.Random(seed); rows=[]
    for i in range(n_groups):
        split=split_for_index(i,n_groups); x0=rng.randint(1,9); a=rng.randint(2,5); b=rng.randint(1,6)
        rule=str((x0+a)*b); wrong=str(x0+a*b)
        if rule==wrong: wrong=str(int(rule)+rng.choice([-3,3]))
        distract=[]
        for d in [1,-1,2,-2,3,-3,5,-5]:
            x=str(int(rule)+d)
            if x not in {rule,wrong} and x not in distract: distract.append(x)
            if len(distract)==2: break
        options,rl,sl=balanced_choice_assignment(rng,rule,wrong,distract)
        code=f"x = {x0}\nx = x + {a}\nx = x * {b}\nWhat is x at the end?"
        opts="\nOptions:\n"+option_block(list(options.keys()),list(options.values()))+"\nReturn only the option letter."
        group=f"code_{seed}_{i:06d}"
        for cond in ["R-only","R+S-conflict"]:
            cue="" if cond=="R-only" else f"\nA quick reader guessed option {sl}."
            rows.append({"sample_id":f"{group}_{cond}","group_id":group,"task_family":"code_trace","template_id":"T0","condition":cond,"split":split,"prompt":code+cue+opts,"correct_answer":rl,"rule_answer":rl,"shortcut_answer":sl,"answer_choices":json_dumps(list(options.keys())),"option_values":json_dumps(options),"reasoning_depth":2,"metadata":json_dumps({"seed":seed})})
    return pd.DataFrame(rows)


def gen_retrieval(n_groups=500, seed=51):
    rng=random.Random(seed); rows=[]
    for i in range(n_groups):
        split=split_for_index(i,n_groups); key=f"K{i:04d}"; true=f"V{rng.randrange(100,999)}"; wrong=f"V{rng.randrange(100,999)}"
        while wrong==true: wrong=f"V{rng.randrange(100,999)}"
        distract=[f"V{rng.randrange(100,999)}" for _ in range(2)]
        while len(set([true,wrong]+distract))<4: distract=[f"V{rng.randrange(100,999)}" for _ in range(2)]
        options,rl,sl=balanced_choice_assignment(rng,true,wrong,distract)
        facts=[f"{key} maps to {true}."]+[f"D{j}_{i} maps to V{rng.randrange(100,999)}." for j in range(8)]
        rng.shuffle(facts)
        opts="\nOptions:\n"+option_block(list(options.keys()),list(options.values()))+"\nReturn only the option letter."
        base="Read the mapping facts and answer the query.\n"+"\n".join(facts)+f"\nQuery: what does {key} map to?"
        group=f"retr_{seed}_{i:06d}"
        for cond in ["R-only","R+S-conflict"]:
            cue="" if cond=="R-only" else f"\nA previous answer selected option {sl}."
            rows.append({"sample_id":f"{group}_{cond}","group_id":group,"task_family":"retrieval","template_id":"T0","condition":cond,"split":split,"prompt":base+cue+opts,"correct_answer":rl,"rule_answer":rl,"shortcut_answer":sl,"answer_choices":json_dumps(list(options.keys())),"option_values":json_dumps(options),"reasoning_depth":1,"metadata":json_dumps({"seed":seed})})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n_groups",type=int,default=500); ap.add_argument("--seed",type=int,default=31); ap.add_argument("--out",required=True)
    a=ap.parse_args(); frames=[gen_graph(a.n_groups,a.seed),gen_arith(a.n_groups,a.seed+1),gen_plan(a.n_groups,a.seed+2),gen_code(a.n_groups,a.seed+3),gen_retrieval(a.n_groups,a.seed+4)]
    df=pd.concat(frames,ignore_index=True); save_table(df,a.out); print(df.groupby(["task_family","condition"]).size()); print(f"saved {len(df)} -> {a.out}")
if __name__=="__main__": main()
