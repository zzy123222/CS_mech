from __future__ import annotations

import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook, make_projection_ablation_hook, make_steer_hook
from src.metrics.generic_health import EXTENDED_PAIRS, generic_nll, generic_nll_with_hook_factory
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_json, save_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed


def margin(bundle,p,r,s,hooks=None):
    z=score_answer_pairs(bundle,[p,p],[str(r),str(s)],batch_size=2,hooks=hooks); return float(z[0]-z[1])


def sample_rows(df,condition,max_samples,seed):
    g=df[df.condition==condition].copy()
    if max_samples and len(g)>max_samples: g=g.sample(max_samples,random_state=seed)
    return g


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--mode",choices=["ablation","steering","rescue","projection"],required=True)
    ap.add_argument("--module",required=True,help="e.g. L18:mlp or L18:resid")
    ap.add_argument("--rescue_layer",type=int,default=None)
    ap.add_argument("--vector",default=None,help=".npy reasoning direction; required for steering/rescue/projection")
    ap.add_argument("--alphas",nargs="*",type=float,default=None)
    ap.add_argument("--mean_condition",default="R-only")
    ap.add_argument("--target_condition",default="R+S-conflict")
    ap.add_argument("--max_samples",type=int,default=128)
    ap.add_argument("--generic_battery",choices=["base","extended"],default="base",
                    help="extended uses the >=100-item battery (deep-interpretation report §8.2)")
    ap.add_argument("--seed",type=int,default=None,help="Override config seed (recorded in outputs)")
    args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(args.seed if args.seed is not None else cfg.get("seed",1)); set_seed(seed); bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    spec=ModuleSpec.parse(args.module); module=resolve_module(bundle.model,spec)
    control=sample_rows(df,args.mean_condition,min(args.max_samples,128),seed)
    control_prompts=[bundle.prepare_prompt(x,enable_thinking=mcfg.get("enable_thinking")) for x in control.prompt]
    target=sample_rows(df,args.target_condition,args.max_samples,seed)
    target_prompts=[bundle.prepare_prompt(x,enable_thinking=mcfg.get("enable_thinking")) for x in target.prompt]
    mean_vec=None
    if args.mode in {"ablation","rescue"}:
        mean_vec=estimate_mean_vector(bundle,control_prompts,spec,max_samples=len(control_prompts))
    vector=None
    if args.mode in {"steering","rescue","projection"}:
        if not args.vector: raise ValueError(f"--vector is required for {args.mode}")
        vector=torch.from_numpy(np.load(args.vector)).float()
    rows=[]
    alphas=args.alphas or (cfg.get("steering_alphas",[-3,-2,-1,-.5,0,.5,1,2,3]) if args.mode=="steering" else [0.5,1.0,1.5])
    for idx,(_,row) in enumerate(tqdm(target.iterrows(),total=len(target),desc=args.mode)):
        p=target_prompts[idx]; base=margin(bundle,p,row.rule_answer,row.shortcut_answer)
        prompt_pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1
        if args.mode=="ablation":
            hooks=[(module,make_mean_ablation_hook(mean_vec,prompt_pos))]; val=margin(bundle,p,row.rule_answer,row.shortcut_answer,hooks)
            rows.append({"sample_id":row.sample_id,"group_id":row.group_id,"module":args.module,"baseline_margin":base,"intervened_margin":val,"effect":val-base})
        elif args.mode=="steering":
            if spec.kind!="resid": raise ValueError("Reasoning vectors from run_probe are residual-stream vectors; steering --module must be :resid")
            for a in alphas:
                val=margin(bundle,p,row.rule_answer,row.shortcut_answer,[(module,make_steer_hook(vector,a,prompt_pos))])
                rows.append({"sample_id":row.sample_id,"group_id":row.group_id,"module":args.module,"alpha":a,"baseline_margin":base,"intervened_margin":val,"effect":val-base})
        elif args.mode=="projection":
            if spec.kind!="resid": raise ValueError("Probe directions are residual-stream vectors; projection --module must be :resid")
            for a in alphas:
                val=margin(bundle,p,row.rule_answer,row.shortcut_answer,[(module,make_projection_ablation_hook(vector,a,prompt_pos))])
                rows.append({"sample_id":row.sample_id,"group_id":row.group_id,"module":args.module,"alpha":a,"baseline_margin":base,"intervened_margin":val,"effect":val-base})
        else:
            if args.rescue_layer is None: raise ValueError("--rescue_layer is required for rescue")
            rescue_spec=ModuleSpec(args.rescue_layer,"resid"); rescue_module=resolve_module(bundle.model,rescue_spec)
            ab_hooks=[(module,make_mean_ablation_hook(mean_vec,prompt_pos))]; ab=margin(bundle,p,row.rule_answer,row.shortcut_answer,ab_hooks)
            for a in alphas:
                hooks=ab_hooks+[(rescue_module,make_steer_hook(vector,a,prompt_pos))]
                resc=margin(bundle,p,row.rule_answer,row.shortcut_answer,hooks)
                rate=(resc-ab)/(base-ab+(1e-8 if base>=ab else -1e-8))
                rows.append({"sample_id":row.sample_id,"group_id":row.group_id,"module":args.module,"rescue_layer":args.rescue_layer,"alpha":a,"baseline_margin":base,"ablated_margin":ab,"rescued_margin":resc,"rescue_rate":rate})
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch"))/"interventions"; out_dir.mkdir(parents=True,exist_ok=True)
    name=f"{args.mode}_{args.module.replace(':','_')}.parquet"; out=pd.DataFrame(rows); save_table(out,out_dir/name)
    battery = EXTENDED_PAIRS if args.generic_battery == "extended" else None
    if args.mode in {"ablation","steering","projection"}:
        g_base=generic_nll(bundle,mcfg,pairs=battery)
        if args.mode in {"ablation",}:
            g_int=generic_nll_with_hook_factory(bundle,mcfg,
                lambda p,pos:[(module,make_mean_ablation_hook(mean_vec,pos))],pairs=battery)
            target_effect=float(out.effect.mean()) if "effect" in out else float("nan")
            summary={"module":args.module,"seed":seed,"battery":args.generic_battery,"generic_nll_base":g_base,
                     "generic_nll_intervened":g_int,"generic_damage":g_int-g_base,
                     "target_effect_mean":target_effect,
                     "specificity_ratio":target_effect/(abs(g_int-g_base)+1e-8)}
            save_json(summary,out_dir/f"generic_ablation_{args.module.replace(':','_')}.json"); print(summary)
        elif args.mode=="projection":
            grows=[]
            for a in sorted(out.alpha.unique()):
                g_int=generic_nll_with_hook_factory(bundle,mcfg,
                    lambda p,pos,a=a:[(module,make_projection_ablation_hook(vector,a,pos))],pairs=battery)
                eff=float(out[out.alpha==a].effect.mean())
                grows.append({"alpha":a,"generic_nll_base":g_base,"generic_nll_intervened":g_int,
                              "generic_damage":g_int-g_base,"target_effect_mean":eff,
                              "specificity_ratio":eff/(abs(g_int-g_base)+1e-8)})
            gdf=pd.DataFrame(grows); save_table(gdf,out_dir/f"generic_projection_{args.module.replace(':','_')}.csv"); print(gdf.to_string(index=False))
        else:
            grows=[]
            for a in sorted(out.alpha.unique()):
                g_int=generic_nll_with_hook_factory(bundle,mcfg,
                    lambda p,pos,a=a:[(module,make_steer_hook(vector,a,pos))],pairs=battery)
                eff=float(out[out.alpha==a].effect.mean())
                grows.append({"alpha":a,"generic_nll_base":g_base,"generic_nll_intervened":g_int,
                              "generic_damage":g_int-g_base,"target_effect_mean":eff,
                              "specificity_ratio":eff/(abs(g_int-g_base)+1e-8)})
            gdf=pd.DataFrame(grows); save_table(gdf,out_dir/f"generic_steering_{args.module.replace(':','_')}.csv"); print(gdf.to_string(index=False))
    print(out.groupby("alpha").mean(numeric_only=True).to_string() if "alpha" in out else out.mean(numeric_only=True).to_string())
if __name__=="__main__": main()
