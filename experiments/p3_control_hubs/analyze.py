from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--result_dir",default="results/p3_control_hubs"); ap.add_argument("--figure_dir",default="figures/p3_control_hubs"); args=ap.parse_args(); r=Path(args.result_dir); f=Path(args.figure_dir); f.mkdir(parents=True,exist_ok=True)
    path=r/"dependency_matrix.csv"
    if path.exists():
        d=pd.read_csv(path,index_col=0); fig,ax=plt.subplots(figsize=(1.8+1.2*d.shape[1],1.8+.45*d.shape[0])); im=ax.imshow(d.values,aspect="auto"); ax.set_xticks(range(d.shape[1]),d.columns,rotation=30,ha="right"); ax.set_yticks(range(d.shape[0]),d.index); ax.set_title("Module × task standardized dependency"); fig.colorbar(im,ax=ax,label="D"); fig.tight_layout(); fig.savefig(f/"dependency_heatmap.pdf"); fig.savefig(f/"dependency_heatmap.png",dpi=300); plt.close(fig)
    epi=r/"epistasis.parquet"
    if epi.exists():
        d=pd.read_parquet(epi); d["pair"]=d.module_A+" + "+d.module_B; top=d.reindex(d.epistasis.abs().sort_values(ascending=False).index).head(20); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(top.pair,top.epistasis,xerr=1.96*top.epistasis_sem); ax.axvline(0,linewidth=.8); ax.set_xlabel("Epistasis = ΔAB − ΔA − ΔB"); fig.tight_layout(); fig.savefig(f/"epistasis_top.pdf"); fig.savefig(f/"epistasis_top.png",dpi=300); plt.close(fig)
    print(f"figures -> {f}")
if __name__=="__main__": main()
