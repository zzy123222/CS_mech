from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.interventions.mediation import downstream_mediation_patch
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import ModuleSpec
from src.utils.seed import set_seed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--source",required=True); ap.add_argument("--downstream",required=True); ap.add_argument("--max_pairs",type=int,default=64); args=ap.parse_args(); cfg=load_yaml(args.config); seed=int(cfg.get("seed",1)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"]); c=df[df.condition=="R-only"].set_index("group_id"); x=df[df.condition=="R+S-conflict"].set_index("group_id"); ids=c.index.intersection(x.index).tolist()[:args.max_pairs]; s=ModuleSpec.parse(args.source); d=ModuleSpec.parse(args.downstream); rows=[]
    for gid in tqdm(ids,desc="mediation"):
        a=c.loc[gid]; b=x.loc[gid]; pa=bundle.prepare_prompt(a.prompt,enable_thinking=mcfg.get("enable_thinking")); pb=bundle.prepare_prompt(b.prompt,enable_thinking=mcfg.get("enable_thinking")); res=downstream_mediation_patch(bundle,pa,pb,str(b.rule_answer),str(b.shortcut_answer),s,d); rows.append({"group_id":gid,"source":args.source,"downstream":args.downstream,**res})
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch"))/"mediation"; out_dir.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); save_table(out,out_dir/f"mediation_{args.source.replace(':','_')}_to_{args.downstream.replace(':','_')}.parquet"); print(out.mean(numeric_only=True).to_string())
if __name__=="__main__": main()
