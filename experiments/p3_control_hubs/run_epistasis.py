from __future__ import annotations

import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed


def margin(bundle,p,r,s,hooks=None):
    z=score_answer_pairs(bundle,[p,p],[str(r),str(s)],batch_size=2,hooks=hooks); return float(z[0]-z[1])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--modules",nargs="+",required=True); ap.add_argument("--task",default=None); ap.add_argument("--max_samples",type=int,default=96); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",31)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"]); ctl=df[df.condition==cfg.get("control_condition","R-only")]
    if len(ctl)>192: ctl=ctl.sample(192,random_state=seed)
    means={}; mods={}
    cps=[bundle.prepare_prompt(p,enable_thinking=mcfg.get("enable_thinking")) for p in ctl.prompt]
    for text in args.modules:
        spec=ModuleSpec.parse(text); mods[text]=resolve_module(bundle.model,spec); means[text]=estimate_mean_vector(bundle,cps,spec,max_samples=len(cps))
    tasks=[args.task] if args.task else sorted(df.task_family.unique()); rows=[]
    for task in tasks:
        tar=df[(df.task_family==task)&(df.condition==cfg.get("target_condition","R+S-conflict"))]
        if len(tar)==0: continue
        if len(tar)>args.max_samples: tar=tar.sample(args.max_samples,random_state=seed)
        for A,B in itertools.combinations(args.modules,2):
            deltas=[]
            for _,row in tqdm(tar.iterrows(),total=len(tar),desc=f"{task} {A}+{B}",leave=False):
                p=bundle.prepare_prompt(row.prompt,enable_thinking=mcfg.get("enable_thinking")); base=margin(bundle,p,row.rule_answer,row.shortcut_answer)
                prompt_pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1; hA=(mods[A],make_mean_ablation_hook(means[A],prompt_pos)); hB=(mods[B],make_mean_ablation_hook(means[B],prompt_pos))
                a=margin(bundle,p,row.rule_answer,row.shortcut_answer,[hA]); b=margin(bundle,p,row.rule_answer,row.shortcut_answer,[hB]); ab=margin(bundle,p,row.rule_answer,row.shortcut_answer,[hA,hB])
                dA=base-a; dB=base-b; dAB=base-ab; epi=dAB-dA-dB; deltas.append((dA,dB,dAB,epi))
            arr=np.asarray(deltas); rows.append({"task_family":task,"module_A":A,"module_B":B,"n":len(arr),"delta_A":arr[:,0].mean(),"delta_B":arr[:,1].mean(),"delta_AB":arr[:,2].mean(),"epistasis":arr[:,3].mean(),"epistasis_sem":arr[:,3].std(ddof=1)/np.sqrt(len(arr)) if len(arr)>1 else np.nan})
    out_dir=Path(cfg.get("output_dir","results/p3_control_hubs")); out_dir.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); save_table(out,out_dir/"epistasis.parquet"); print(out.sort_values("epistasis",key=lambda s:s.abs(),ascending=False).to_string(index=False))
if __name__=="__main__": main()
