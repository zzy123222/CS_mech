from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from src.datasets.common import parse_choices
from src.metrics.behavior import reasoning_reliance_index, cohens_d
from src.models.scoring import score_answer_pairs, score_choice_matrix
from src.stats.bootstrap import bootstrap_mean
from src.utils.experiment import load_bundle_from_model_config, prepare_prompts
from src.utils.io import load_yaml, load_table, save_table, save_json, sha256_file
from src.utils.metadata import write_run_metadata
from src.utils.seed import set_seed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--max_rows",type=int,default=None); ap.add_argument("--thinking",choices=["true","false","config"],default="config"); ap.add_argument("--seed",type=int,default=None,help="Override config seed (recorded in metadata)"); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(args.seed if args.seed is not None else cfg.get("seed",1)); set_seed(seed)
    bundle,mcfg=load_bundle_from_model_config(cfg["model_config"])
    df=load_table(cfg["dataset"])
    if args.max_rows: df=df.head(args.max_rows).copy()
    enable=mcfg.get("enable_thinking")
    if args.thinking!="config": enable=(args.thinking=="true")
    prompt_texts=prepare_prompts(bundle,df.prompt.tolist(),enable_thinking=enable)
    bs=int(cfg.get("batch_size",8))
    df=df.copy(); df["prompt_text"]=prompt_texts
    df["logp_rule"]=score_answer_pairs(bundle,prompt_texts,df.rule_answer.astype(str).tolist(),batch_size=bs)
    df["logp_shortcut"]=score_answer_pairs(bundle,prompt_texts,df.shortcut_answer.astype(str).tolist(),batch_size=bs)
    df["margin"]=df.logp_rule-df.logp_shortcut
    if cfg.get("score_all_choices",True):
        choices=[parse_choices(x) for x in df.answer_choices]
        mat=score_choice_matrix(bundle,prompt_texts,choices,batch_size=bs)
        df["pred_answer"]=[opts[int(np.argmax(row))] for opts,row in zip(choices,mat)]
        df["is_correct"]=(df.pred_answer.astype(str)==df.correct_answer.astype(str)).astype(int)
        for j in range(mat.shape[1]): df[f"logp_choice_{j}"]=mat[:,j]
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch")); out_dir.mkdir(parents=True,exist_ok=True)
    save_table(df,out_dir/"behavior.parquet")
    summary=[]
    for cond,g in df.groupby("condition"):
        ci=bootstrap_mean(g.margin.values,n_boot=2000,seed=seed)
        summary.append({"condition":cond,"n":len(g),"mean_margin":g.margin.mean(),"margin_ci_low":ci["ci_low"],"margin_ci_high":ci["ci_high"],"accuracy":float(g.is_correct.mean()) if "is_correct" in g else np.nan})
    if {"R+S-conflict","S-only"}.issubset(set(df.condition)):
        rri=reasoning_reliance_index(df)
    else: rri=np.nan
    save_table(pd.DataFrame(summary),out_dir/"behavior_summary.csv")
    save_json({"RRI":rri},out_dir/"rri.json")
    write_run_metadata(out_dir/"run_metadata_behavior.json",model_id=bundle.model_id,revision=bundle.revision,seed=seed,config_file=args.config,dataset_hash=sha256_file(cfg["dataset"]),extra={"enable_thinking":enable})
    print(pd.DataFrame(summary).to_string(index=False)); print(f"RRI={rri:.4f}")

if __name__=="__main__": main()
