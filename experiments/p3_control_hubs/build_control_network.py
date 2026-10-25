from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dependency",default="results/p3_control_hubs/dependency_matrix.parquet"); ap.add_argument("--epistasis",default="results/p3_control_hubs/epistasis.parquet"); ap.add_argument("--effect_threshold",type=float,default=.5); ap.add_argument("--epi_threshold",type=float,default=.2); ap.add_argument("--out",default="results/p3_control_hubs/control_network.graphml"); args=ap.parse_args()
    try: import networkx as nx
    except ImportError as e: raise ImportError("pip install networkx") from e
    dep=pd.read_parquet(args.dependency); G=nx.Graph()
    for m,g in dep.groupby("module"):
        G.add_node(m,kind="module",pleiotropy=float(g.pleiotropy_index.iloc[0]) if "pleiotropy_index" in g else float('nan'))
        for _,r in g.iterrows():
            task=f"task::{r.task_family}"; G.add_node(task,kind="task");
            if abs(float(r.D_standardized))>=args.effect_threshold: G.add_edge(m,task,edge_type="dependency",weight=float(r.D_standardized))
    ep=Path(args.epistasis)
    if ep.exists():
        e=pd.read_parquet(ep)
        for _,r in e.iterrows():
            if abs(float(r.epistasis))>=args.epi_threshold: G.add_edge(r.module_A,r.module_B,edge_type="epistasis",weight=float(r.epistasis),task=str(r.task_family))
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); nx.write_graphml(G,out); print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} -> {out}")
if __name__=="__main__": main()
