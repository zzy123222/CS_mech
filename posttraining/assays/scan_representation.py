from __future__ import annotations
import sys, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
import torch

from src.interventions.patching import capture_module_vector
from src.utils.io import load_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed

SITES = [("L20", ModuleSpec(20, "resid"), "monitor"),
         ("L22mlp", ModuleSpec(22, "mlp"), "controller"),
         ("L24attn", ModuleSpec(24, "attn"), "integrator"),
         ("L25attn", ModuleSpec(25, "attn"), "opponent")]


def load_merged(base_id, adapter_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


class Bundle:
    def __init__(self, model, tok):
        self.model = model
        self.tokenizer = tok

    def prepare_prompt(self, prompt, enable_thinking=None):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def cv_auc(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    aucs = []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[f], y[f]
        d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        s = Xte @ d
        order = np.argsort(s)
        ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        pos = yte == 1
        n1, n0 = pos.sum(), (~pos).sum()
        if n1 == 0 or n0 == 0:
            continue
        aucs.append((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(aucs)) if aucs else np.nan


def collect(bundle, model, prompts, spec):
    vecs = []
    for p in prompts:
        v = capture_module_vector(bundle, p, spec, position=-1)
        vecs.append(v.float().cpu().numpy())
    return np.stack(vecs)


def scan_model(model, tok, df_p2, tag, n, rows, ref_dirs=None):
    bundle = Bundle(model, tok)
    from src.datasets.generate_error_monitor import build_check_prompt
    rows_p2 = df_p2.head(n)
    err_p, cln_p = [], []
    for _, row in rows_p2.iterrows():
        t = max(1, min(int(row.n_steps), int(row.error_step) + 1))
        err_p.append(bundle.prepare_prompt(build_check_prompt(row, "error", t), enable_thinking=False))
        cln_p.append(bundle.prepare_prompt(build_check_prompt(row, "clean", t), enable_thinking=False))

    for site, spec, mech in SITES:
        Xe = collect(bundle, model, err_p, spec)
        Xc = collect(bundle, model, cln_p, spec)
        auc = cv_auc(np.concatenate([Xe, Xc]),
                     np.concatenate([np.ones(len(Xe)), np.zeros(len(Xc))]))
        d = Xe.mean(0) - Xc.mean(0)
        align = (float(np.dot(d, ref_dirs[site]) /
                       (np.linalg.norm(d) * np.linalg.norm(ref_dirs[site]) + 1e-9))
                 if ref_dirs is not None else np.nan)
        rows.append({"model": tag, "site": site, "mechanism": mech,
                     "probe_auc": round(auc, 4),
                     "dir_align_vs_instruct": round(align, 4) if ref_dirs is not None else None})
        print(f"  {tag} {site}: AUC={auc:.3f} align={align if ref_dirs is None else round(align,3)}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", type=int, default=[0, 100, 200, 500, 1000])
    ap.add_argument("--arms", nargs="*", default=["general_sft_seed1", "correction_sft_seed1",
                                                   "sham_correction_seed1"])
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="posttraining/results/formal/representation_index.csv")
    args = ap.parse_args()

    set_seed(1)
    df_p2 = load_table("data/generated/p2_errors.parquet")

    rows = []
    model, tok = load_merged("Qwen/Qwen3-4B", None)
    ref_rows = []
    scan_model(model, tok, df_p2, "instruct_ref", args.n, ref_rows, ref_dirs=None)
    ref_dirs = {}
    bundle = Bundle(model, tok)
    from src.datasets.generate_error_monitor import build_check_prompt
    rows_p2 = df_p2.head(args.n)
    err_p = [bundle.prepare_prompt(build_check_prompt(r, "error", max(1, min(int(r.n_steps), int(r.error_step) + 1))), enable_thinking=False)
             for _, r in rows_p2.iterrows()]
    cln_p = [bundle.prepare_prompt(build_check_prompt(r, "clean", max(1, min(int(r.n_steps), int(r.error_step) + 1))), enable_thinking=False)
             for _, r in rows_p2.iterrows()]
    for site, spec, mech in SITES:
        Xe = collect(bundle, model, err_p, spec)
        Xc = collect(bundle, model, cln_p, spec)
        ref_dirs[site] = Xe.mean(0) - Xc.mean(0)
    rows.extend(ref_rows)
    del model, bundle
    gc.collect(); torch.cuda.empty_cache()

    for arm in args.arms:
        for step in args.steps:
            ckpt = ROOT / "posttraining/results/formal" / arm / f"step_{step:04d}"
            if not ckpt.exists():
                print(f"[skip] {ckpt}")
                continue
            model, tok = load_merged(args.base_model, str(ckpt))
            scan_model(model, tok, df_p2, f"{arm}@{step}", args.n, rows, ref_dirs=ref_dirs)
            del model
            gc.collect(); torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
