from __future__ import annotations

import numpy as np


def best_piecewise_linear_change(x, y, min_segment=3):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<2*min_segment: return {"index":None,"x":None,"sse":np.nan}
    best=(None,np.inf)
    for k in range(min_segment,len(x)-min_segment+1):
        sse=0.0
        for xs,ys in [(x[:k],y[:k]),(x[k:],y[k:])]:
            coef=np.polyfit(xs,ys,1); pred=np.polyval(coef,xs); sse+=float(((ys-pred)**2).sum())
        if sse<best[1]: best=(k,sse)
    k,sse=best; return {"index":int(k),"x":float(x[k]),"sse":float(sse)}


def pelt_change_points(y, penalty=3.0):
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError("Install optional dependency 'ruptures' for PELT") from e
    arr=np.asarray(y,float).reshape(-1,1); return rpt.Pelt(model="rbf").fit(arr).predict(pen=penalty)
