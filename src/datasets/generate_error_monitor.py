from __future__ import annotations

import argparse
import json
import random
import pandas as pd

from src.utils.io import save_table


def _make_trace(rng: random.Random, n_steps: int):
    start = rng.randint(10, 50)
    ops, vals, v = [], [start], start
    for _ in range(n_steps):
        op = rng.choice(["+", "-", "*"])
        if op == "+": k=rng.randint(2,9); v += k
        elif op == "-": k=rng.randint(2,min(9,max(2,v-1))); v -= k
        else: k=rng.choice([2,3]); v *= k
        ops.append((op,k)); vals.append(v)
    return start, ops, vals


def render_prefix(start, ops, vals, upto, override=None):
    lines=[f"Start value: {start}."]
    for j in range(upto):
        op,k=ops[j]; val = vals[j+1] if override is None or j not in override else override[j]
        lines.append(f"Step {j+1}: apply {op}{k}; result = {val}.")
    return "\n".join(lines)


def _inject_error(rng: random.Random, kind: str, prev: int, correct: int):
    if kind == "sign":
        return -correct
    if kind == "carry":
        if abs(correct) >= 10:
            return correct + rng.choice([-10, 10])
        return correct + rng.choice([-2, 2])
    return correct + rng.choice([-2, -1, 1, 2])


def generate(n_samples=2000, seed=11, n_steps=4, error_types=None) -> pd.DataFrame:
    if isinstance(error_types, str):
        error_types = [t.strip() for t in error_types.split(",") if t.strip()]
    error_types = error_types or ["offset"]
    unknown = [t for t in error_types if t not in {"offset", "sign", "carry"}]
    if unknown:
        raise ValueError(f"unknown error types: {unknown}")
    rng=random.Random(seed); rows=[]
    for i in range(n_samples):
        start, ops, vals=_make_trace(rng,n_steps)
        error_step=rng.randrange(1,n_steps)
        error_type=error_types[i % len(error_types)]
        wrong=_inject_error(rng,error_type,vals[error_step],vals[error_step+1])
        if wrong==vals[error_step+1]: wrong=vals[error_step+1]+1
        override={error_step:wrong}
        err_vals=vals.copy(); err_vals[error_step+1]=wrong; v=wrong
        for j in range(error_step+1,n_steps):
            op,k=ops[j]; v = v+k if op=="+" else (v-k if op=="-" else v*k); err_vals[j+1]=v
        rows.append({
            "sample_id":f"err_{seed}_{i:06d}", "task_family":"error_monitor", "start":start,
            "ops_json":json.dumps(ops), "clean_values_json":json.dumps(vals), "error_values_json":json.dumps(err_vals),
            "error_step":error_step+1, "n_steps":n_steps, "error_type":error_type,
            "final_correct":vals[-1], "final_error":err_vals[-1],
        })
    return pd.DataFrame(rows)


def build_check_prompt(row, variant: str, time_step: int) -> str:
    ops=[tuple(x) for x in json.loads(row["ops_json"])]
    clean_vals=json.loads(row["clean_values_json"]); err_vals=json.loads(row["error_values_json"])
    vals = clean_vals if variant=="clean" else err_vals
    prefix=render_prefix(int(row["start"]), ops, vals, time_step)
    return ("A solver is executing a deterministic arithmetic trace. Inspect the trace so far.\n" + prefix +
            "\nShould the solver CONTINUE, or CHECK the work because an earlier step is wrong? Answer only CHECK or CONTINUE.")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n_samples",type=int,default=2000); ap.add_argument("--seed",type=int,default=11); ap.add_argument("--n_steps",type=int,default=4)
    ap.add_argument("--error_types",default="offset",help="comma list of offset,sign,carry; multiple types enable cross-type transfer probes")
    ap.add_argument("--out",required=True)
    a=ap.parse_args(); df=generate(a.n_samples,a.seed,a.n_steps,error_types=a.error_types); save_table(df,a.out); print(f"saved {len(df)} rows -> {a.out}"); print(df.groupby("error_type").size())
if __name__=="__main__": main()
