from __future__ import annotations

import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import ModuleSpec, get_decoder_layers, resolve_module
from src.utils.seed import set_seed


def margin(bundle,p,r,s,hooks=None):
    z=score_answer_pairs(bundle,[p,p],[str(r),str(s)],batch_size=2,hooks=hooks); return float(z[0]-z[1])


def ablation_effect(bundle,mcfg,spec,ctl_prompts,target_rows):
    module=resolve_module(bundle.model,spec)
    mean_vec=estimate_mean_vector(bundle,ctl_prompts,spec,max_samples=len(ctl_prompts))
    effects=[]
    for row,p in target_rows:
        pos=len(bundle.tokenizer(p,add_special_tokens=False)["input_ids"])-1
        hook=(module,make_mean_ablation_hook(mean_vec,pos))
        base=margin(bundle,p,row.rule_answer,row.shortcut_answer)
        abl=margin(bundle,p,row.rule_answer,row.shortcut_answer,[hook])
        effects.append(base-abl)
    return float(np.mean(effects))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--candidate",required=True,help="e.g. L18:mlp")
    ap.add_argument("--n_random",type=int,default=20)
    ap.add_argument("--layer_band",type=int,default=6,help="Sample null modules within +/- this many layers")
    ap.add_argument("--mean_condition",default="R-only")
    ap.add_argument("--target_condition",default="R+S-conflict")
    ap.add_argument("--max_samples",type=int,default=48)
    ap.add_argument("--seed",type=int,default=None)
    args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(args.seed if args.seed is not None else cfg.get("seed",1)); set_seed(seed)
    bundle,mcfg=load_bundle_from_model_config(cfg["model_config"]); df=load_table(cfg["dataset"])
    cand=ModuleSpec.parse(args.candidate); n_layers=len(get_decoder_layers(bundle.model))
    rng=random.Random(seed)
    lo,hi=max(0,cand.layer-args.layer_band),min(n_layers-1,cand.layer+args.layer_band)
    pool=[ModuleSpec(l,cand.kind) for l in range(lo,hi+1) if ModuleSpec(l,cand.kind)!=cand]
    if args.n_random<=len(pool):
        null_specs=rng.sample(pool,args.n_random); note=f"sampled {args.n_random} of {len(pool)}"
    else:
        null_specs=pool; note=f"exhaustive band ({len(pool)} modules); bootstrap B={args.n_random}"
    ctl=df[df.condition==args.mean_condition].head(min(args.max_samples,128))
    ctl_prompts=[bundle.prepare_prompt(p,enable_thinking=mcfg.get("enable_thinking")) for p in ctl.prompt]
    tar=df[df.condition==args.target_condition].head(args.max_samples)
    target_rows=[(row,bundle.prepare_prompt(row.prompt,enable_thinking=mcfg.get("enable_thinking")) ) for _,row in tar.iterrows()]
    rows=[]
    cand_effect=ablation_effect(bundle,mcfg,cand,ctl_prompts,target_rows)
    rows.append({"module":args.candidate,"is_candidate":True,"effect":cand_effect})
    for spec in tqdm(null_specs,desc="null modules"):
        eff=ablation_effect(bundle,mcfg,spec,ctl_prompts,target_rows)
        rows.append({"module":f"L{spec.layer}:{spec.kind}","is_candidate":False,"effect":eff})
    out=pd.DataFrame(rows)
    null=out[~out.is_candidate].effect.values
    p_emp=float((1+np.sum(null>=cand_effect))/(1+len(null)))
    boot=np.asarray([float(np.mean(np.random.default_rng(seed+i).choice(null,len(null),replace=True))) for i in range(min(max(args.n_random,1000),10000))])
    boot_ci=(float(np.quantile(boot,.025)),float(np.quantile(boot,.975)))
    out_dir=Path(cfg.get("output_dir","results/p1_reasoning_switch"))/"controls"; out_dir.mkdir(parents=True,exist_ok=True)
    save_table(out,out_dir/f"random_module_control_{args.candidate.replace(':','_')}.csv")
    save_table(pd.DataFrame([{"candidate":args.candidate,"candidate_effect":cand_effect,
                              "null_mean":float(null.mean()),"null_sd":float(null.std(ddof=1)),
                              "null_max":float(null.max()),"n_null":len(null),"protocol":note,
                              "null_mean_boot_ci_low":boot_ci[0],"null_mean_boot_ci_high":boot_ci[1],
                              "empirical_p_one_sided":p_emp}]),
               out_dir/f"random_module_control_{args.candidate.replace(':','_')}_summary.csv")
    print(out.sort_values("effect",ascending=False).to_string(index=False))
    print(f"candidate={cand_effect:.4f} null(mean±sd)={null.mean():.4f}±{null.std(ddof=1):.4f} "
          f"boot95%CI=[{boot_ci[0]:.3f},{boot_ci[1]:.3f}] p={p_emp:.4g} ({note})")
if __name__=="__main__": main()
