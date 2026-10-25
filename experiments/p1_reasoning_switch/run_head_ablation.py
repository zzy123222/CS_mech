from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.interventions.heads import estimate_head_mean, make_head_mean_ablation_prehook
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import get_decoder_layers, get_attention_output_projection, get_num_attention_heads
from src.utils.seed import set_seed


def margin(bundle,p,r,s,prehook=None,proj=None):
    handle=proj.register_forward_pre_hook(prehook) if prehook else None
    try:
        z=score_answer_pairs(bundle,[p,p],[str(r),str(s)],batch_size=2); return float(z[0]-z[1])
    finally:
        if handle: handle.remove()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--layer",type=int,required=True); ap.add_argument("--heads",nargs="+",type=int,required=True); ap.add_argument("--mean_condition",default="R-only"); ap.add_argument("--target_condition",default="R+S-conflict"); ap.add_argument("--max_samples",type=int,default=128); ap.add_argument("--split_filter",default=None,choices=["train","val","test_ood"],help="Restrict samples to a pre-registered split (discovery vs confirmatory, report #4 §4)"); ap.add_argument("--tag",default="",help="Filename suffix, e.g. _discovery"); ap.add_argument("--fdr",action="store_true",help="Report BH-FDR q-values over heads"); args=ap.parse_args(); cfg=load_yaml(args.config); seed=int(cfg.get("seed",1)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    if args.split_filter:
        if "split" not in df.columns: raise SystemExit("dataset has no split column")
        df=df[df.split==args.split_filter].copy()
    layer=get_decoder_layers(bundle.model)[args.layer]; proj=get_attention_output_projection(layer); n_heads=get_num_attention_heads(bundle.model,layer)
    ctl=df[df.condition==args.mean_condition].head(min(args.max_samples,128)); tar=df[df.condition==args.target_condition].head(args.max_samples); cps=[bundle.prepare_prompt(p,enable_thinking=mcfg.get("enable_thinking")) for p in ctl.prompt]; rows=[]
    for head in args.heads:
        if head<0 or head>=n_heads: raise ValueError(f"head {head} outside [0,{n_heads})")
        mean=estimate_head_mean(bundle,cps,args.layer,head,max_samples=len(cps))
        for _,row in tqdm(tar.iterrows(),total=len(tar),desc=f"L{args.layer}H{head}",leave=False):
            p=bundle.prepare_prompt(row.prompt,enable_thinking=mcfg.get("enable_thinking")); prompt_pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1; pre=make_head_mean_ablation_prehook(mean,head,n_heads,prompt_pos); base=margin(bundle,p,row.rule_answer,row.shortcut_answer); abl=margin(bundle,p,row.rule_answer,row.shortcut_answer,pre,proj); rows.append({"sample_id":row.sample_id,"layer":args.layer,"head":head,"baseline_margin":base,"ablated_margin":abl,"damage":base-abl})
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch"))/"heads"; out_dir.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); save_table(out,out_dir/f"head_ablation_L{args.layer:03d}{args.tag}.parquet")
    sm=out.groupby(["layer","head"]).damage.agg(["mean","sem","count"]).sort_values("mean",ascending=False).reset_index()
    if args.fdr and len(sm)>1:
        from scipy import stats as st
        from src.stats.fdr import bh_fdr
        t=sm["mean"]/(sm["sem"]+1e-12); p=2*(1-st.norm.cdf(abs(t))); sm["p"]=p; sm["q"]=bh_fdr(p.tolist())
    print(sm.to_string(index=False))
if __name__=="__main__": main()
