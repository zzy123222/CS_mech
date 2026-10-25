from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook
from src.metrics.behavior import pleiotropy_index
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed

GENERIC_PAIRS=[
    ("The capital of France is", " Paris"),
    ("2 + 2 =", " 4"),
    ("Complete the sequence: A B C", " D"),
    ("The opposite of hot is", " cold"),
]

def margin(bundle,p,r,s,hooks=None):
    z=score_answer_pairs(bundle,[p,p],[str(r),str(s)],batch_size=2,hooks=hooks); return float(z[0]-z[1])

def generic_damage_score(bundle, mcfg, module, mean_vec):
    base_nll=[]; abl_nll=[]
    for raw,ans in GENERIC_PAIRS:
        p=bundle.prepare_prompt(raw,enable_thinking=mcfg.get("enable_thinking"))
        token_n=max(1,len(bundle.tokenizer(ans,add_special_tokens=False)["input_ids"]))
        b=score_answer_pairs(bundle,[p],[ans],batch_size=1)[0]
        pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1
        hook=(module,make_mean_ablation_hook(mean_vec,pos))
        a=score_answer_pairs(bundle,[p],[ans],batch_size=1,hooks=[hook])[0]
        base_nll.append(-b/token_n); abl_nll.append(-a/token_n)
    return float(np.mean(abl_nll)-np.mean(base_nll))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--modules",nargs="*",default=None); ap.add_argument("--max_samples_per_task",type=int,default=None); ap.add_argument("--mean_scope",choices=["task","global"],default="task"); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",31)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    modules=args.modules or cfg.get("candidate_modules",[]); control_cond=cfg.get("control_condition","R-only"); target_cond=cfg.get("target_condition","R+S-conflict"); maxn=args.max_samples_per_task or int(cfg.get("max_samples_per_task",128))
    tasks=sorted(df.task_family.unique()); rows=[]
    global_control=df[df.condition==control_cond]
    if len(global_control)>256: global_control=global_control.sample(256,random_state=seed)
    for module_text in modules:
        spec=ModuleSpec.parse(module_text); module=resolve_module(bundle.model,spec)
        global_mean=None
        if args.mean_scope=="global":
            cps=[bundle.prepare_prompt(p,enable_thinking=mcfg.get("enable_thinking")) for p in global_control.prompt]; global_mean=estimate_mean_vector(bundle,cps,spec,max_samples=len(cps))
        module_task_effects=[]
        for task in tasks:
            ctl=df[(df.task_family==task)&(df.condition==control_cond)]
            tar=df[(df.task_family==task)&(df.condition==target_cond)]
            if len(tar)==0 or len(ctl)==0: continue
            if len(tar)>maxn: tar=tar.sample(maxn,random_state=seed)
            if len(ctl)>min(maxn,128): ctl=ctl.sample(min(maxn,128),random_state=seed)
            mean_vec=global_mean
            if mean_vec is None:
                cps=[bundle.prepare_prompt(p,enable_thinking=mcfg.get("enable_thinking")) for p in ctl.prompt]; mean_vec=estimate_mean_vector(bundle,cps,spec,max_samples=len(cps))
            base=[]; abl=[]
            generic_damage=generic_damage_score(bundle,mcfg,module,mean_vec)
            for _,row in tqdm(tar.iterrows(),total=len(tar),desc=f"{module_text} {task}",leave=False):
                p=bundle.prepare_prompt(row.prompt,enable_thinking=mcfg.get("enable_thinking")); prompt_pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1; hook=(module,make_mean_ablation_hook(mean_vec,prompt_pos)); base.append(margin(bundle,p,row.rule_answer,row.shortcut_answer)); abl.append(margin(bundle,p,row.rule_answer,row.shortcut_answer,[hook]))
            base=np.asarray(base); abl=np.asarray(abl); sd=base.std(ddof=1) if len(base)>1 else 1.0; D=float((base.mean()-abl.mean())/(sd+1e-8)); module_task_effects.append(D)
            rows.append({"module":module_text,"task_family":task,"n":len(base),"baseline_margin_mean":base.mean(),"ablated_margin_mean":abl.mean(),"effect_raw":base.mean()-abl.mean(),"D_standardized":D,"generic_damage":generic_damage,"specificity_ratio":(base.mean()-abl.mean())/(abs(generic_damage)+1e-8) if np.isfinite(generic_damage) else np.nan})
        pi=pleiotropy_index(module_task_effects)
        for r in rows:
            if r["module"]==module_text: r["pleiotropy_index"]=pi
    out_dir=Path(cfg.get("output_dir","results/p3_control_hubs")); out_dir.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); save_table(out,out_dir/"dependency_matrix.parquet"); pivot=out.pivot(index="module",columns="task_family",values="D_standardized"); pivot.to_csv(out_dir/"dependency_matrix.csv"); print(out.sort_values(["module","D_standardized"],ascending=[True,False]).to_string(index=False))
if __name__=="__main__": main()
