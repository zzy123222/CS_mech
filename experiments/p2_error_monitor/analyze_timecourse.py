from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.probes.linear_probe import fit_binary_probe, centroid_direction
from src.utils.io import save_table


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--activation_dir",required=True); ap.add_argument("--out",default=None); ap.add_argument("--seed",type=int,default=11); args=ap.parse_args()
    out_dir=Path(args.out or (Path(args.activation_dir).parent/"analysis")); out_dir.mkdir(parents=True,exist_ok=True); rows=[]; xrows=[]
    for path in sorted(Path(args.activation_dir).glob("layer_*.pt")):
        obj=torch.load(path,map_location="cpu",weights_only=False); H=obj["H"].float().numpy(); m=pd.DataFrame(obj["meta"]); lid=int(obj["layer"])
        for t in sorted(m.time_step.unique()):
            mask=(m.time_step==t); mt=m.loc[mask].reset_index(drop=True); X=H[mask.values]; y=(mt.variant=="error").astype(int).values
            post=(mt.is_post_error==1).values; X=X[post]; y=y[post]; groups=mt.loc[post,"sample_id"].values; etypes=mt.loc[post,"error_type"].values if "error_type" in mt else np.array(["offset"]*len(y))
            if len(np.unique(y))<2 or len(y)<30: continue
            gss=GroupShuffleSplit(n_splits=1,test_size=.3,random_state=args.seed); tr,te=next(gss.split(X,y,groups))
            res=fit_binary_probe(X[tr],y[tr],X[te],y[te],seed=args.seed); pos=X[tr][y[tr]==1]; neg=X[tr][y[tr]==0]; v=centroid_direction(pos,neg); np.save(out_dir/f"error_direction_L{lid:03d}_t{t}.npy",v.astype(np.float32))
            rows.append({"layer":lid,"time_step":int(t),"auc":res["auc"],"accuracy":res["accuracy"],"n":len(y)})
            types=sorted(set(etypes))
            for src_t in types:
                for tgt_t in types:
                    if src_t==tgt_t: continue
                    tr_m=etypes==src_t; te_m=etypes==tgt_t
                    if tr_m.sum()<15 or te_m.sum()<15 or len(np.unique(y[tr_m]))<2 or len(np.unique(y[te_m]))<2: continue
                    res_t=fit_binary_probe(X[tr_m],y[tr_m],X[te_m],y[te_m],seed=args.seed)
                    xrows.append({"layer":lid,"time_step":int(t),"train_type":src_t,"test_type":tgt_t,"auc_transfer":res_t["auc"],"n_train":int(tr_m.sum()),"n_test":int(te_m.sum())})
    out=pd.DataFrame(rows); save_table(out,out_dir/"error_probe_timecourse.csv"); print(out.sort_values("auc",ascending=False).head(30).to_string(index=False))
    if xrows:
        x=pd.DataFrame(xrows); save_table(x,out_dir/"error_probe_cross_type.csv"); print("\ncross-type transfer:\n",x.to_string(index=False))
if __name__=="__main__": main()
