from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.stats.change_point import best_piecewise_linear_change
from src.utils.io import load_table, save_table, save_json


def first_threshold_step(d, threshold):
    q=d.sort_values("step"); hit=q[q.value>=threshold]; return int(hit.step.iloc[0]) if len(hit) else None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="results/p4_development/checkpoint_scan.parquet"); ap.add_argument("--out_dir",default="results/p4_development"); ap.add_argument("--figure_dir",default="figures/p4_development"); args=ap.parse_args(); df=load_table(args.input); out_dir=Path(args.out_dir); fig_dir=Path(args.figure_dir); out_dir.mkdir(parents=True,exist_ok=True); fig_dir.mkdir(parents=True,exist_ok=True)
    summaries=[]
    curves={}
    for metric in ["RRI","probe_auc","patch_recovery"]:
        d=df[df.metric==metric].copy()
        if len(d)==0: continue
        if metric in {"probe_auc","patch_recovery"}: d=d.groupby("step",as_index=False).value.max()
        else: d=d.groupby("step",as_index=False).value.mean()
        curves[metric]=d.sort_values("step")
        cp=best_piecewise_linear_change(np.log1p(d.step.values),d.value.values,min_segment=max(2,min(3,len(d)//3)))
        final=float(d.value.iloc[-1]); th50=(0.5 if metric=="probe_auc" else 0.0)+0.5*(final-(0.5 if metric=="probe_auc" else 0.0)); th80=(0.5 if metric=="probe_auc" else 0.0)+0.8*(final-(0.5 if metric=="probe_auc" else 0.0))
        summaries.append({"metric":metric,"change_step":None if cp["index"] is None else int(d.step.iloc[cp["index"]]),"threshold50":th50,"step50":first_threshold_step(d,th50),"threshold80":th80,"step80":first_threshold_step(d,th80)})
    sm=pd.DataFrame(summaries); save_table(sm,out_dir/"development_summary.csv")
    lookup={r.metric:r.step50 for r in sm.itertuples()}; internal=[x for x in [lookup.get("probe_auc"),lookup.get("patch_recovery")] if x is not None]; behavior=lookup.get("RRI"); gap=(behavior-min(internal)) if behavior is not None and internal else None; save_json({"latent_emergence_gap_step50":gap},out_dir/"latent_emergence_gap.json")
    fig,ax=plt.subplots(figsize=(8,4.5))
    for metric,d in curves.items(): ax.plot(d.step,d.value,marker="o",label=metric)
    ax.set_xscale("symlog",linthresh=1); ax.set_xlabel("Training step"); ax.set_ylabel("Normalized / raw metric"); ax.legend(); fig.tight_layout(); fig.savefig(fig_dir/"development_curves.pdf"); fig.savefig(fig_dir/"development_curves.png",dpi=300); plt.close(fig)
    print(sm.to_string(index=False)); print("latent_emergence_gap_step50=",gap)
if __name__=="__main__": main()
