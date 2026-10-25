from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.hooks import make_steer_hook
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed


def orthogonal_randoms(v: torch.Tensor, n: int, seed: int):
    g=torch.Generator(device="cpu"); g.manual_seed(seed); v=v.float(); v=v/(v.norm()+1e-8); out=[]
    for _ in range(n):
        r=torch.randn(v.shape,generator=g); r=r-torch.dot(r,v)*v; r=r/(r.norm()+1e-8); out.append(r)
    return out


def margin(bundle,p,r,s,hooks=None):
    z=score_answer_pairs(bundle,[p,p],[str(r),str(s)],batch_size=2,hooks=hooks); return float(z[0]-z[1])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--layer",type=int,required=True); ap.add_argument("--vector",required=True); ap.add_argument("--alpha",type=float,default=1.0); ap.add_argument("--n_random",type=int,default=50); ap.add_argument("--max_samples",type=int,default=96); args=ap.parse_args(); cfg=load_yaml(args.config); seed=int(cfg.get("seed",1)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"]); tar=df[df.condition=="R+S-conflict"].head(args.max_samples); module=resolve_module(bundle.model,ModuleSpec(args.layer,"resid")); v=torch.from_numpy(np.load(args.vector)).float(); controls=orthogonal_randoms(v,args.n_random,seed); rows=[]
    vectors=[("reasoning",0,v)]+[("random_orthogonal",i+1,r) for i,r in enumerate(controls)]
    for kind,vid,vec in vectors:
        effects=[]
        for _,row in tqdm(tar.iterrows(),total=len(tar),desc=f"{kind}{vid}",leave=False):
            p=bundle.prepare_prompt(row.prompt,enable_thinking=mcfg.get("enable_thinking")); pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1; base=margin(bundle,p,row.rule_answer,row.shortcut_answer); val=margin(bundle,p,row.rule_answer,row.shortcut_answer,[(module,make_steer_hook(vec,args.alpha,pos))]); effects.append(val-base)
        rows.append({"vector_type":kind,"vector_id":vid,"mean_effect":float(np.mean(effects)),"sem":float(np.std(effects,ddof=1)/np.sqrt(len(effects))) if len(effects)>1 else np.nan,"n":len(effects)})
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch"))/"controls"; out_dir.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); save_table(out,out_dir/f"steering_controls_L{args.layer:03d}.csv"); rand=out[out.vector_type=="random_orthogonal"].mean_effect.values; true=float(out.iloc[0].mean_effect); p_emp=(1+np.sum(rand>=true))/(1+len(rand)); print(out.head().to_string(index=False)); print(f"empirical one-sided p={p_emp:.4g}")
if __name__=="__main__": main()
