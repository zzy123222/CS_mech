from __future__ import annotations

import numpy as np


def bh_fdr(pvalues, alpha: float | None = None):
    p=np.asarray(pvalues,float)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    n=len(p)
    if n == 0:
        q=np.asarray([],dtype=float)
        return (q, np.asarray([],dtype=bool)) if alpha is not None else q
    order=np.argsort(p); ranked=p[order]
    q=ranked*n/np.arange(1,n+1)
    q=np.minimum.accumulate(q[::-1])[::-1]
    q=np.clip(q,0,1)
    out=np.empty_like(q); out[order]=q
    if alpha is None:
        return out
    return out, out <= float(alpha)
