from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.interventions.patching import capture_module_vector, margin_with_hook
from src.interventions.hooks import make_patch_hook
from src.stats.bootstrap import bootstrap_mean
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, parse_int_list, save_table
from src.utils.landmarks import LANDMARKS, landmark_token_position
from src.utils.module_access import ModuleSpec, get_decoder_layers, resolve_module
from src.utils.seed import set_seed


def paired_rows(df, clean_cond, corrupt_cond, max_pairs=None, seed=0):
    c=df[df.condition==clean_cond].set_index("group_id"); x=df[df.condition==corrupt_cond].set_index("group_id"); ids=c.index.intersection(x.index).tolist()
    if max_pairs and len(ids)>max_pairs:
        import random; rng=random.Random(seed); ids=rng.sample(ids,max_pairs)
    return [(gid,c.loc[gid],x.loc[gid]) for gid in ids]


def summarize(out, min_denom, seed):
    if "recovery" not in out:
        return None, None
    ok = out.dropna(subset=["recovery"]).copy()
    ok["denom"] = ok.clean_margin - ok.corrupt_margin
    kept = ok[ok.denom.abs() >= min_denom]
    def _agg(df):
        rows=[]
        for (lid,kind,pos),g in df.groupby(["layer","kind","position"]):
            ci=bootstrap_mean(g.recovery.values,n_boot=2000,seed=seed)
            rows.append({"layer":lid,"kind":kind,"position":pos,"mean_recovery":ci["mean"],
                         "ci_low":ci["ci_low"],"ci_high":ci["ci_high"],"n":len(g),
                         "denom_median":float(g.denom.abs().median())})
        return pd.DataFrame(rows).sort_values("mean_recovery",ascending=False)
    return _agg(kept), _agg(ok)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--layers",nargs="*",default=["all"])
    ap.add_argument("--kind",choices=["resid","attn","mlp"],default="resid")
    ap.add_argument("--max_pairs",type=int,default=None)
    ap.add_argument("--position",type=int,default=None,help="Legacy single absolute/negative token index; overrides --positions")
    ap.add_argument("--positions",nargs="*",default=["last"],
                    help=f"Semantic landmarks: {' '.join(LANDMARKS)}; resolved per prompt on both clean and corrupt sides")
    ap.add_argument("--min_denom",type=float,default=0.25,
                    help="Drop pairs with |clean-corrupt| below this from the main summary (log-prob units)")
    ap.add_argument("--seed",type=int,default=None,help="Override config seed")
    args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(args.seed if args.seed is not None else cfg.get("seed",1)); set_seed(seed)
    bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    p_cfg=cfg.get("patching",{}); clean_cond=p_cfg.get("clean_condition","R-only"); corrupt_cond=p_cfg.get("corrupt_condition","R+S-conflict"); max_pairs=args.max_pairs or p_cfg.get("max_pairs",256)
    pairs=paired_rows(df,clean_cond,corrupt_cond,max_pairs=max_pairs,seed=seed)
    n_layers=len(get_decoder_layers(bundle.model)); layers=list(range(n_layers)) if args.layers==["all"] else [i for x in args.layers for i in parse_int_list(x)]
    if args.position is not None:
        pos_names, pos_mode = ["fixed"], "int"
    else:
        unknown=[p for p in args.positions if p not in LANDMARKS]
        if unknown: raise SystemExit(f"Unknown positions {unknown}; choose from {LANDMARKS}")
        pos_names, pos_mode = args.positions, "landmark"
    rows=[]
    for gid,clean,corrupt in tqdm(pairs,desc="pairs"):
        clean_prompt=bundle.prepare_prompt(clean.prompt,enable_thinking=mcfg.get("enable_thinking")); corrupt_prompt=bundle.prepare_prompt(corrupt.prompt,enable_thinking=mcfg.get("enable_thinking"))
        r,s=str(corrupt.rule_answer),str(corrupt.shortcut_answer)
        clean_margin=margin_with_hook(bundle,clean_prompt,r,s)
        corrupt_margin=margin_with_hook(bundle,corrupt_prompt,r,s)
        denom=clean_margin-corrupt_margin
        eps=1e-8 if denom>=0 else -1e-8
        for pname in pos_names:
            if pos_mode=="int":
                sp=tp=args.position
            else:
                sp=landmark_token_position(bundle.tokenizer,clean_prompt,clean.prompt,pname)
                tp=landmark_token_position(bundle.tokenizer,corrupt_prompt,corrupt.prompt,pname)
            for lid in layers:
                spec=ModuleSpec(lid,args.kind)
                try:
                    module=resolve_module(bundle.model,spec)
                    src=capture_module_vector(bundle,clean_prompt,spec,sp)
                    patched=margin_with_hook(bundle,corrupt_prompt,r,s,hooks=[(module,make_patch_hook(src,target_pos=tp))])
                    rows.append({"group_id":gid,"layer":lid,"kind":args.kind,"position":pname,
                                 "clean_margin":clean_margin,"corrupt_margin":corrupt_margin,
                                 "patched_margin":patched,"recovery":(patched-corrupt_margin)/(denom+eps)})
                except Exception as e:
                    rows.append({"group_id":gid,"layer":lid,"kind":args.kind,"position":pname,"error":repr(e)})
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch"))/"patching"; out_dir.mkdir(parents=True,exist_ok=True)
    out=pd.DataFrame(rows); save_table(out,out_dir/f"patching_{args.kind}.parquet")
    sm_f,sm_u=summarize(out,args.min_denom,seed)
    if sm_f is not None:
        save_table(sm_f,out_dir/f"patching_{args.kind}_summary.csv")
        save_table(sm_u,out_dir/f"patching_{args.kind}_summary_unfiltered.csv")
        print(sm_f.head(20).to_string(index=False))
if __name__=="__main__": main()
