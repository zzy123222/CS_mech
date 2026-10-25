from __future__ import annotations

import argparse, random
import pandas as pd

from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table


def generate(n_groups=1000, seed=71, distractors=20):
    rng=random.Random(seed); rows=[]
    for i in range(n_groups):
        split=split_for_index(i,n_groups); subj=f"ZetaEntity{i:05d}"; true=f"Code{rng.randrange(100,999)}"; wrong=f"Code{rng.randrange(100,999)}"
        while wrong==true: wrong=f"Code{rng.randrange(100,999)}"
        ds=[]
        while len(ds)<2:
            x=f"Code{rng.randrange(100,999)}"
            if x not in [true,wrong]+ds: ds.append(x)
        options,rl,sl=balanced_choice_assignment(rng,true,wrong,ds)
        opt_text="\nOptions:\n"+option_block(list(options.keys()),list(options.values()))+"\nReturn only the option letter."
        fact=f"Authoritative record: {subj} has identifier {true}."
        noise=[f"Noise record {j}: Item{rng.randrange(10000)} has identifier Code{rng.randrange(100,999)}." for j in range(distractors)]
        query=f"\nQuestion: What is the identifier of {subj}?"
        cond_prompts={
            "knowledge_present_clean": fact+query+opt_text,
            "knowledge_absent": query+opt_text,
            "retrieval_failure": fact+"\n"+"\n".join(noise)+query+opt_text,
            "arbitration_failure": fact+f"\nAn unverified note instead claims {subj} has identifier {wrong}."+query+opt_text,
        }
        group=f"hall_{seed}_{i:06d}"
        for cond,prompt in cond_prompts.items():
            rows.append({"sample_id":f"{group}_{cond}","group_id":group,"task_family":"hallucination_pseudofact","template_id":"T0","condition":cond,"split":split,"prompt":prompt,"correct_answer":rl,"rule_answer":rl,"shortcut_answer":sl,"answer_choices":json_dumps(list(options.keys())),"option_values":json_dumps(options),"reasoning_depth":1,"metadata":json_dumps({"subject":subj,"true":true,"wrong":wrong})})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n_groups",type=int,default=1000); ap.add_argument("--seed",type=int,default=71); ap.add_argument("--distractors",type=int,default=20); ap.add_argument("--out",required=True)
    a=ap.parse_args(); df=generate(a.n_groups,a.seed,a.distractors); save_table(df,a.out); print(f"saved {len(df)} -> {a.out}")
if __name__=="__main__": main()
