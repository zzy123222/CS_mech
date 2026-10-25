from __future__ import annotations

import numpy as np


def paired_bootstrap(x, y, n_boot=5000, seed=0):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)!=len(y): raise ValueError("paired_bootstrap requires equal lengths")
    d=x-y; rng=np.random.default_rng(seed); n=len(d)
    means=np.empty(n_boot)
    for b in range(n_boot): means[b]=d[rng.integers(0,n,n)].mean()
    return {"mean_diff":float(d.mean()),"ci_low":float(np.quantile(means,.025)),"ci_high":float(np.quantile(means,.975)),"p_two_sided":float(2*min((means<=0).mean(),(means>=0).mean()))}


def bootstrap_mean(x, n_boot=5000, seed=0):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); n=len(x); means=np.empty(n_boot)
    for b in range(n_boot): means[b]=x[rng.integers(0,n,n)].mean()
    return {"mean":float(x.mean()),"ci_low":float(np.quantile(means,.025)),"ci_high":float(np.quantile(means,.975))}
