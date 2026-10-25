from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch

from src.datasets.generate_error_monitor import build_check_prompt
from src.hooks.batched import collect_last_token_activations
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config, prepare_prompts
from src.utils.io import load_yaml, load_table, parse_int_list, save_table
from src.utils.module_access import get_decoder_layers
from src.utils.seed import set_seed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--layers",nargs="*",default=["all"]); ap.add_argument("--max_samples",type=int,default=None); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",11)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    if args.max_samples or cfg.get("max_samples"): df=df.head(args.max_samples or int(cfg["max_samples"])).copy()
    nl=len(get_decoder_layers(bundle.model)); layers=list(range(nl)) if args.layers==["all"] else [i for x in args.layers for i in parse_int_list(x)]
    records=[]; activation_chunks={lid:[] for lid in layers}; meta=[]
    max_t=int(df.n_steps.max())
    for t in range(1,max_t+1):
        for variant in ["clean","error"]:
            sub=df.copy(); raw=[build_check_prompt(row,variant,t) for _,row in sub.iterrows()]; prompts=prepare_prompts(bundle,raw,enable_thinking=mcfg.get("enable_thinking"))
            check=[str(cfg.get("check_answer","CHECK"))]*len(sub); cont=[str(cfg.get("continue_answer","CONTINUE"))]*len(sub)
            sp=score_answer_pairs(bundle,prompts,check,batch_size=int(cfg.get("batch_size",4))); sn=score_answer_pairs(bundle,prompts,cont,batch_size=int(cfg.get("batch_size",4))); margins=sp-sn
            acts=collect_last_token_activations(bundle,prompts,layers,batch_size=int(cfg.get("batch_size",4)))
            for j,(_,row) in enumerate(sub.iterrows()):
                et=row.get("error_type","offset")
                records.append({"sample_id":row.sample_id,"variant":variant,"time_step":t,"error_step":int(row.error_step),"check_margin":float(margins[j]),"is_post_error":int(t>=int(row.error_step)),"error_type":et})
                meta.append({"sample_id":row.sample_id,"variant":variant,"time_step":t,"error_step":int(row.error_step),"is_post_error":int(t>=int(row.error_step)),"error_type":et})
            for lid in layers: activation_chunks[lid].append(acts[lid])
    out_dir=Path(cfg.get("output_dir","results/p2_error_monitor")); out_dir.mkdir(parents=True,exist_ok=True); save_table(pd.DataFrame(records),out_dir/"timecourse_behavior.parquet")
    act_dir=out_dir/"activations"; act_dir.mkdir(exist_ok=True)
    for lid in layers: torch.save({"layer":lid,"H":torch.cat(activation_chunks[lid]).to(torch.float16),"meta":meta},act_dir/f"layer_{lid:03d}.pt")
    print(f"saved timecourse for {len(df)} samples, {max_t} steps, {len(layers)} layers")
if __name__=="__main__": main()
