from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.io import load_table


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--result_dir",default="results/p1_reasoning_switch"); ap.add_argument("--figure_dir",default="figures/p1_reasoning_switch"); args=ap.parse_args()
    r=Path(args.result_dir); f=Path(args.figure_dir); f.mkdir(parents=True,exist_ok=True)
    bpath=r/"behavior.parquet"
    if bpath.exists():
        b=load_table(bpath); g=b.groupby("condition").margin.agg(["mean","sem"]).reset_index();
        fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(g.condition,g["mean"],yerr=1.96*g["sem"],capsize=4); ax.axhline(0,linewidth=.8); ax.set_ylabel("Rule − shortcut sequence log-prob margin"); ax.tick_params(axis='x',rotation=20); fig.tight_layout(); fig.savefig(f/"fig1_behavior_margin.pdf"); fig.savefig(f/"fig1_behavior_margin.png",dpi=300); plt.close(fig)
    p=r/"probe"/"probe_summary.csv"
    if p.exists():
        d=pd.read_csv(p); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(d.layer,d.auc,marker="o"); ax.axhline(.5,linestyle="--",linewidth=.8); ax.set(xlabel="Layer",ylabel="OOD probe AUC"); fig.tight_layout(); fig.savefig(f/"fig2_probe_auc.pdf"); fig.savefig(f/"fig2_probe_auc.png",dpi=300); plt.close(fig)
    pp=r/"patching"/"patching_resid_summary.csv"
    if pp.exists():
        d=pd.read_csv(pp).sort_values("layer"); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(d.layer,d.mean_recovery,marker="o"); ax.fill_between(d.layer,d.ci_low,d.ci_high,alpha=.2); ax.axhline(0,linewidth=.8); ax.set(xlabel="Layer",ylabel="Normalized patch recovery"); fig.tight_layout(); fig.savefig(f/"fig3_patching_recovery.pdf"); fig.savefig(f/"fig3_patching_recovery.png",dpi=300); plt.close(fig)
    print(f"figures written to {f}")
if __name__=="__main__": main()
