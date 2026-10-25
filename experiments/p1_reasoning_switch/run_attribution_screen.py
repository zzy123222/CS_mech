from __future__ import annotations

import argparse, random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.interventions.attribution import residual_attribution_patching
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.seed import set_seed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--max_pairs",type=int,default=64); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",1)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    c=df[df.condition=="R-only"].set_index("group_id"); x=df[df.condition=="R+S-conflict"].set_index("group_id"); ids=c.index.intersection(x.index).tolist(); random.Random(seed).shuffle(ids); ids=ids[:args.max_pairs]
    rows=[]
    for gid in tqdm(ids,desc="attribution pairs"):
        clean=c.loc[gid]; corrupt=x.loc[gid]
        clean_prompt=bundle.prepare_prompt(clean.prompt,enable_thinking=mcfg.get("enable_thinking")); corrupt_prompt=bundle.prepare_prompt(corrupt.prompt,enable_thinking=mcfg.get("enable_thinking"))
        try:
            local=residual_attribution_patching(bundle,clean_prompt,corrupt_prompt,str(corrupt.rule_answer),str(corrupt.shortcut_answer),position=-1)
            for r in local: rows.append({"group_id":gid,**r})
        except ValueError as e:
            raise RuntimeError("Attribution screening requires rule/shortcut labels to tokenize as one token. Run the token audit and use stable labels A/B/C/D.") from e
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch"))/"patching"; out_dir.mkdir(parents=True,exist_ok=True)
    out=pd.DataFrame(rows); save_table(out,out_dir/"attribution_resid.parquet"); sm=out.groupby("layer").attribution_patch.agg(["mean","std","count"]).reset_index().sort_values("mean",ascending=False); save_table(sm,out_dir/"attribution_resid_summary.csv"); print(sm.head(20).to_string(index=False))
if __name__=="__main__": main()
