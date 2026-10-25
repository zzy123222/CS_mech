from __future__ import annotations

import numpy as np
import pandas as pd


def add_margin(df: pd.DataFrame, rule_col="logp_rule", shortcut_col="logp_shortcut") -> pd.DataFrame:
    out=df.copy(); out["margin"]=out[rule_col]-out[shortcut_col]; return out


def group_paired_contrast(df: pd.DataFrame, cond_a: str, cond_b: str,
                          value: str = "margin", key: str = "group_id") -> pd.DataFrame:
    a = df[df.condition == cond_a].set_index(key)
    b = df[df.condition == cond_b].set_index(key)
    ids = a.index.intersection(b.index)
    out = pd.DataFrame({
        "group_id": ids,
        "a": a.loc[ids, value].astype(float).values,
        "b": b.loc[ids, value].astype(float).values,
    })
    out["diff"] = out.a - out.b
    return out


def reasoning_reliance_index(df: pd.DataFrame, conflict="R+S-conflict", shortcut="S-only") -> float:
    a=df.loc[df.condition==conflict,"margin"].mean(); b=df.loc[df.condition==shortcut,"margin"].mean(); return float(a-b)


def cohens_d(x, y, paired: bool = False) -> float:
    x=np.asarray(x,float); y=np.asarray(y,float)
    if paired:
        d=x-y; sd=d.std(ddof=1); return float(d.mean()/sd) if sd>0 else np.nan
    nx,ny=len(x),len(y); pooled=np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/max(nx+ny-2,1)); return float((x.mean()-y.mean())/pooled) if pooled>0 else np.nan


def normalized_recovery(clean, corrupt, patched, eps=1e-8):
    clean=np.asarray(clean,float); corrupt=np.asarray(corrupt,float); patched=np.asarray(patched,float)
    den=clean-corrupt; return (patched-corrupt)/(den+np.where(den>=0,eps,-eps))


def rescue_rate(baseline, ablated, rescued, eps=1e-8):
    baseline=np.asarray(baseline,float); ablated=np.asarray(ablated,float); rescued=np.asarray(rescued,float)
    den=baseline-ablated; return (rescued-ablated)/(den+np.where(den>=0,eps,-eps))


def pleiotropy_index(effects, eps=1e-12) -> float:
    x=np.abs(np.asarray(effects,float)); T=len(x)
    if T==0 or x.sum()<=eps: return np.nan
    p=x/x.sum(); h=-(p*np.log(p+eps)).sum(); return float(np.exp(h)/T)
