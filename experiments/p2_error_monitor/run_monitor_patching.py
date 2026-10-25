from __future__ import annotations

import argparse, random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.datasets.generate_error_monitor import build_check_prompt
from src.interventions.patching import patch_margin
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, parse_int_list, save_table
from src.utils.module_access import ModuleSpec, get_decoder_layers
from src.utils.seed import set_seed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--layers",nargs="*",default=["all"]); ap.add_argument("--direction",choices=["error_to_clean","clean_to_error"],default="error_to_clean"); ap.add_argument("--time_offset",type=int,default=0,help="Evaluate at error_step + offset"); ap.add_argument("--max_samples",type=int,default=128); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",11)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    if len(df)>args.max_samples: df=df.sample(args.max_samples,random_state=seed)
    nl=len(get_decoder_layers(bundle.model)); layers=list(range(nl)) if args.layers==["all"] else [i for x in args.layers for i in parse_int_list(x)]
    rows=[]
    for _,row in tqdm(df.iterrows(),total=len(df),desc="monitor patch"):
        t=max(1,min(int(row.n_steps),int(row.error_step)+args.time_offset)); clean=build_check_prompt(row,"clean",t); error=build_check_prompt(row,"error",t)
        clean=bundle.prepare_prompt(clean,enable_thinking=mcfg.get("enable_thinking")); error=bundle.prepare_prompt(error,enable_thinking=mcfg.get("enable_thinking"))
        source,target=(error,clean) if args.direction=="error_to_clean" else (clean,error)
        for lid in layers:
            res=patch_margin(bundle,source,target,str(cfg.get("check_answer","CHECK")),str(cfg.get("continue_answer","CONTINUE")),ModuleSpec(lid,"resid"),-1,-1)
            rows.append({"sample_id":row.sample_id,"layer":lid,"time_step":t,"direction":args.direction,**res})
    out_dir=Path(cfg.get("output_dir","results/p2_error_monitor"))/"patching"; out_dir.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); save_table(out,out_dir/f"monitor_{args.direction}.parquet"); sm=out.groupby("layer").recovery.agg(["mean","sem","count"]).reset_index().sort_values("mean",ascending=False); save_table(sm,out_dir/f"monitor_{args.direction}_summary.csv"); print(sm.head(20).to_string(index=False))
if __name__=="__main__": main()
