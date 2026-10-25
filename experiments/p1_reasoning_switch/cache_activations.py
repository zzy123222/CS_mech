from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch

from src.hooks.batched import collect_last_token_activations
from src.utils.experiment import load_bundle_from_model_config, prepare_prompts
from src.utils.io import load_yaml, load_table, parse_int_list
from src.utils.module_access import get_decoder_layers
from src.utils.seed import set_seed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--layers",nargs="*",default=None); ap.add_argument("--conditions",nargs="*",default=None); ap.add_argument("--max_rows",type=int,default=None,help="Total cap, sampled uniformly across the filtered rows"); ap.add_argument("--max_per_condition",type=int,default=None,help="Per-condition cap, sampled uniformly within each condition (keeps OOD groups reachable)"); ap.add_argument("--batch_size",type=int,default=None); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",1)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"])
    df=load_table(cfg["dataset"])
    conds=args.conditions or cfg.get("activation_conditions")
    if conds: df=df[df.condition.isin(conds)].copy()
    if args.max_per_condition:
        df=pd.concat([g.sample(min(len(g),args.max_per_condition),random_state=seed) for _,g in df.groupby("condition")],ignore_index=True)
    if args.max_rows and args.max_rows<len(df):
        df=df.sample(args.max_rows,random_state=seed).reset_index(drop=True)
    n_layers=len(get_decoder_layers(bundle.model))
    if not args.layers or args.layers==["all"]: layers=list(range(n_layers))
    else:
        layers=[]
        for x in args.layers:
            layers.extend(parse_int_list(x))
    prompts=prepare_prompts(bundle,df.prompt.tolist(),enable_thinking=mcfg.get("enable_thinking"))
    acts=collect_last_token_activations(bundle,prompts,layers,batch_size=args.batch_size or int(cfg.get("batch_size",8)))
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch"))/"activations"; out_dir.mkdir(parents=True,exist_ok=True)
    meta_cols=[c for c in ["sample_id","group_id","task_family","template_id","condition","split","rule_answer","shortcut_answer"] if c in df]
    meta=df[meta_cols].reset_index(drop=True).to_dict(orient="list")
    for lid,H in acts.items():
        torch.save({"layer":lid,"H":H.to(torch.float16),"meta":meta},out_dir/f"layer_{lid:03d}.pt")
    print(f"saved {len(layers)} layers x {len(df)} examples -> {out_dir}")
if __name__=="__main__": main()
