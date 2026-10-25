from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.generate_error_monitor import build_check_prompt
from src.interventions.hooks import make_steer_hook
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed


def margin(bundle,p,pos,neg,hooks=None):
    z=score_answer_pairs(bundle,[p,p],[pos,neg],batch_size=2,hooks=hooks); return float(z[0]-z[1])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--layer",type=int,required=True); ap.add_argument("--direction_template",required=True,help="path like analysis/error_direction_L018_t{t}.npy"); ap.add_argument("--alpha",type=float,default=1.0); ap.add_argument("--max_samples",type=int,default=128); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",11)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    if len(df)>args.max_samples: df=df.sample(args.max_samples,random_state=seed)
    module=resolve_module(bundle.model,ModuleSpec(args.layer,"resid")); pos=str(cfg.get("check_answer","CHECK")); neg=str(cfg.get("continue_answer","CONTINUE")); rows=[]
    for t in range(1,int(df.n_steps.max())+1):
        path=args.direction_template.format(t=t)
        if not Path(path).exists(): continue
        v=torch.from_numpy(np.load(path)).float()
        for _,row in tqdm(df.iterrows(),total=len(df),desc=f"t={t}",leave=False):
            variant="error" if t>=int(row.error_step) else "clean"; raw=build_check_prompt(row,variant,t); p=bundle.prepare_prompt(raw,enable_thinking=mcfg.get("enable_thinking")); base=margin(bundle,p,pos,neg); prompt_pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1; steered=margin(bundle,p,pos,neg,[(module,make_steer_hook(v,args.alpha,prompt_pos))]); rows.append({"sample_id":row.sample_id,"time_step":t,"error_step":int(row.error_step),"baseline_check_margin":base,"steered_check_margin":steered,"effect":steered-base})
    out_dir=Path(cfg.get("output_dir","results/p2_error_monitor"))/"steering"; out_dir.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); save_table(out,out_dir/f"window_L{args.layer:03d}_a{args.alpha}.parquet"); print(out.groupby("time_step").effect.agg(["mean","sem","count"]).to_string())
if __name__=="__main__": main()
