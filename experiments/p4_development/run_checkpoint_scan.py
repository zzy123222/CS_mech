from __future__ import annotations

import argparse, gc, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.hooks.batched import collect_last_token_activations
from src.interventions.patching import patch_margin
from src.metrics.behavior import reasoning_reliance_index
from src.models.load import load_causal_lm
from src.models.scoring import score_answer_pairs
from src.probes.linear_probe import fit_binary_probe
from src.utils.io import load_yaml, load_table, save_table
from src.utils.module_access import get_decoder_layers, ModuleSpec
from src.utils.seed import set_seed


def step_number(revision):
    m=re.search(r"(\d+)",str(revision)); return int(m.group(1)) if m else 143000


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--revisions",nargs="*",default=None); ap.add_argument("--metrics",nargs="*",default=["behavior","representation","circuit"]); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=int(cfg.get("seed",1)); set_seed(seed); df=load_table(cfg["dataset"]); revisions=args.revisions or cfg["revisions"]; maxn=int(cfg.get("max_samples",256)); out=[]
    if len(df)>maxn*4: df=df.groupby("condition",group_keys=False).head(maxn).copy()
    for revision in revisions:
        print(f"\n=== {revision} ===")
        bundle=load_causal_lm(cfg["model_id"],revision=revision,dtype=cfg.get("dtype","bfloat16"),device_map=cfg.get("device_map","auto"),chat=bool(cfg.get("chat",False)))
        n_layers=len(get_decoder_layers(bundle.model)); layer_ids=sorted(set(min(n_layers-1,max(0,int(round(frac*(n_layers-1))))) for frac in cfg.get("layer_fractions",[.25,.5,.75])))
        base_record={"revision":revision,"step":step_number(revision),"n_layers":n_layers}
        if "behavior" in args.metrics:
            prompts=[bundle.prepare_prompt(p) for p in df.prompt]; r=score_answer_pairs(bundle,prompts,df.rule_answer.astype(str).tolist(),batch_size=8); s=score_answer_pairs(bundle,prompts,df.shortcut_answer.astype(str).tolist(),batch_size=8); tmp=df[["condition"]].copy(); tmp["margin"]=r-s
            base_record["RRI"]=reasoning_reliance_index(tmp) if {"R+S-conflict","S-only"}.issubset(set(tmp.condition)) else np.nan
            base_record["conflict_margin"]=float(tmp.loc[tmp.condition=="R+S-conflict","margin"].mean())
        if "representation" in args.metrics:
            sub=df[df.condition.isin(["R-only","S-only"])].copy(); prompts=[bundle.prepare_prompt(p) for p in sub.prompt]; acts=collect_last_token_activations(bundle,prompts,layer_ids,batch_size=8); y=(sub.condition=="R-only").astype(int).values
            tr=(sub["split"]=="train").values if "split" in sub else np.arange(len(sub))%2==0; te=(sub["split"]=="test_ood").values if "split" in sub else ~tr
            for lid,H in acts.items():
                if tr.sum()>10 and te.sum()>10:
                    res=fit_binary_probe(H.numpy()[tr],y[tr],H.numpy()[te],y[te],seed=seed); out.append({**base_record,"metric":"probe_auc","layer":lid,"layer_fraction":lid/max(n_layers-1,1),"value":res["auc"]})
        if "circuit" in args.metrics:
            clean=df[df.condition=="R-only"].set_index("group_id"); corrupt=df[df.condition=="R+S-conflict"].set_index("group_id"); ids=clean.index.intersection(corrupt.index).tolist(); random.Random(seed).shuffle(ids); ids=ids[:int(cfg.get("patch_pairs",32))]
            for lid in layer_ids:
                rec=[]
                for gid in ids:
                    a=clean.loc[gid]; b=corrupt.loc[gid]; res=patch_margin(bundle,bundle.prepare_prompt(a.prompt),bundle.prepare_prompt(b.prompt),str(b.rule_answer),str(b.shortcut_answer),ModuleSpec(lid,"resid")); rec.append(res["recovery"])
                out.append({**base_record,"metric":"patch_recovery","layer":lid,"layer_fraction":lid/max(n_layers-1,1),"value":float(np.mean(rec))})
        if "behavior" in args.metrics:
            out.append({**base_record,"metric":"RRI","layer":np.nan,"layer_fraction":np.nan,"value":base_record.get("RRI",np.nan)})
            out.append({**base_record,"metric":"conflict_margin","layer":np.nan,"layer_fraction":np.nan,"value":base_record.get("conflict_margin",np.nan)})
        del bundle; gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        out_dir=Path(cfg.get("output_dir","results/p4_development")); out_dir.mkdir(parents=True,exist_ok=True); save_table(pd.DataFrame(out),out_dir/"checkpoint_scan.parquet")
    print(pd.DataFrame(out).tail(30).to_string(index=False))
if __name__=="__main__": main()
