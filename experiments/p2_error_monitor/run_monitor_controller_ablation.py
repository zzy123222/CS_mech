from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.generate_error_monitor import build_check_prompt
from src.hooks.core import first_tensor, temporary_hooks
from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed


def measure(bundle, prompt, check_ans, cont_ans, candidate_hook, readout_module, error_direction, prompt_pos):
    holder={}
    def readout_hook(_m,_inp,output):
        h=first_tensor(output); holder["h"]=h[:,prompt_pos,:].detach().float().cpu()[0]
    hooks=[]
    if candidate_hook is not None: hooks.append(candidate_hook)
    hooks.append((readout_module,readout_hook))
    with temporary_hooks(hooks):
        z=score_answer_pairs(bundle,[prompt,prompt],[check_ans,cont_ans],batch_size=2)
    e=float(torch.dot(holder["h"],error_direction.float()).item())
    return float(z[0]-z[1]), e


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--modules",nargs="+",required=True); ap.add_argument("--readout_layer",type=int,required=True); ap.add_argument("--error_direction",required=True); ap.add_argument("--time_offset",type=int,default=0); ap.add_argument("--max_samples",type=int,default=96); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",11)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    if len(df)>args.max_samples: df=df.sample(args.max_samples,random_state=seed)
    readout_module=resolve_module(bundle.model,ModuleSpec(args.readout_layer,"resid")); edir=torch.from_numpy(np.load(args.error_direction)).float(); check=str(cfg.get("check_answer","CHECK")); cont=str(cfg.get("continue_answer","CONTINUE")); rows=[]
    for mod_text in args.modules:
        spec=ModuleSpec.parse(mod_text); module=resolve_module(bundle.model,spec)
        clean_prompts=[]
        for _,row in df.iterrows():
            t=max(1,min(int(row.n_steps),int(row.error_step)+args.time_offset)); raw=build_check_prompt(row,"clean",t); clean_prompts.append(bundle.prepare_prompt(raw,enable_thinking=mcfg.get("enable_thinking")))
        mean_vec=estimate_mean_vector(bundle,clean_prompts,spec,max_samples=len(clean_prompts))
        for _,row in tqdm(df.iterrows(),total=len(df),desc=mod_text,leave=False):
            t=max(1,min(int(row.n_steps),int(row.error_step)+args.time_offset)); raw=build_check_prompt(row,"error",t); p=bundle.prepare_prompt(raw,enable_thinking=mcfg.get("enable_thinking")); prompt_pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1
            base_c,base_e=measure(bundle,p,check,cont,None,readout_module,edir,prompt_pos)
            ab_hook=(module,make_mean_ablation_hook(mean_vec,prompt_pos)); ab_c,ab_e=measure(bundle,p,check,cont,ab_hook,readout_module,edir,prompt_pos)
            rows.append({"sample_id":row.sample_id,"module":mod_text,"time_step":t,"baseline_correction_margin":base_c,"ablated_correction_margin":ab_c,"delta_C":ab_c-base_c,"baseline_error_rep":base_e,"ablated_error_rep":ab_e,"delta_E":ab_e-base_e})
    out_dir=Path(cfg.get("output_dir","results/p2_error_monitor"))/"controller"; out_dir.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); save_table(out,out_dir/"monitor_controller_ablation.parquet"); sm=out.groupby("module")[["delta_C","delta_E"]].agg(["mean","sem","count"]); sm.to_csv(out_dir/"monitor_controller_summary.csv"); print(sm.to_string())
if __name__=="__main__": main()
