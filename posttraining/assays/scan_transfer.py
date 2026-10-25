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
from src.interventions.steering import mean_direction
from src.interventions.hooks import make_steer_hook
from src.models.scoring import score_answer_pairs
from src.utils.io import load_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed

SPEC = ModuleSpec(24, "attn")


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

    def margin(self, p, r, s, hooks=None):
        z = score_answer_pairs(self, [p, p], [str(r), str(s)], batch_size=2, hooks=hooks)
        return float(z[0] - z[1])


def orthogonal(v, rng):
    w = torch.randn(v.shape, generator=rng, device=v.device, dtype=v.dtype)
    w = w - (w @ v) / (v @ v + 1e-9) * v
    return w / (w.norm() + 1e-9) * v.norm()


def scan_model(model, tok, df_graph, df_retrieval, tag, n, alpha, rows):
    bundle = Bundle(model, tok)
    mod = resolve_module(model, SPEC)

    def prompts(df, cond):
        sub = df[df.condition == cond].head(n)
        return [bundle.prepare_prompt(r.prompt, enable_thinking=False) for _, r in sub.iterrows()], sub

    g_cln_p, _ = prompts(df_graph, "R-only")
    g_cfl_p, g_cfl = prompts(df_graph, "R+S-conflict")
    r_cln_p, _ = prompts(df_retrieval, "R-only")
    r_cfl_p, r_cfl = prompts(df_retrieval, "R+S-conflict")

    def direction(cln, cfl):
        H_cfl = torch.stack([capture_module_vector(bundle, p, SPEC, position=-1) for p in cfl[:32]])
        H_cln = torch.stack([capture_module_vector(bundle, p, SPEC, position=-1) for p in cln[:32]])
        return mean_direction(H_cfl, H_cln).to(torch.bfloat16)

    g_dir = direction(g_cln_p, g_cfl_p)
    r_dir = direction(r_cln_p, r_cfl_p)
    rng = torch.Generator(device="cpu").manual_seed(42)
    o_dir = orthogonal(g_dir.float().cpu(), rng).to(g_dir.device, g_dir.dtype)

    def inject_effect(cfl_p, cfl_sub, vec, sign=+1.0):
        deltas = []
        for (p, (_, row)) in zip(cfl_p, cfl_sub.iterrows()):
            pos = len(bundle.tokenizer(p, add_special_tokens=False)["input_ids"]) - 1
            hook = (mod, make_steer_hook(sign * vec, alpha, pos))
            base = bundle.margin(p, row.rule_answer, row.shortcut_answer)
            steered = bundle.margin(p, row.rule_answer, row.shortcut_answer, hooks=[hook])
            deltas.append(steered - base)
        return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

    m_g2r, lo, hi = inject_effect(r_cfl_p, r_cfl, g_dir)
    rows.append({"model": tag, "effect": "G->R", "mean": round(m_g2r, 4),
                 "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    m_r2g, lo, hi = inject_effect(g_cfl_p, g_cfl, r_dir)
    rows.append({"model": tag, "effect": "R->G", "mean": round(m_r2g, 4),
                 "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    m_orth, lo, hi = inject_effect(r_cfl_p, r_cfl, o_dir)
    rows.append({"model": tag, "effect": "orthogonal_control", "mean": round(m_orth, 4),
                 "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    rows.append({"model": tag, "effect": "asymmetry_G2R_minus_R2G",
                 "mean": round(m_g2r - m_r2g, 4), "ci_low": None, "ci_high": None})
    print(f"  {tag}: G->R {m_g2r:+.3f} | R->G {m_r2g:+.3f} | orth {m_orth:+.3f} "
          f"| A {m_g2r - m_r2g:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", type=int, default=[0, 200, 500, 1000])
    ap.add_argument("--arms", nargs="*", default=["general_sft_seed1", "correction_sft_seed1",
                                                   "sham_correction_seed1"])
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="posttraining/results/formal/transfer_longitudinal.csv")
    args = ap.parse_args()

    set_seed(1)
    battery = load_table("data/generated/p3_task_battery.parquet")
    df_graph = battery[battery.task_family == "graph_reasoning"]
    df_retrieval = battery[battery.task_family == "retrieval"]

    rows = []
    model, tok = load_merged("Qwen/Qwen3-4B", None)
    scan_model(model, tok, df_graph, df_retrieval, "instruct_ref", args.n, args.alpha, rows)
    del model
    gc.collect(); torch.cuda.empty_cache()

    for arm in args.arms:
        for step in args.steps:
            ckpt = ROOT / "posttraining/results/formal" / arm / f"step_{step:04d}"
            if not ckpt.exists():
                print(f"[skip] {ckpt}")
                continue
            model, tok = load_merged(args.base_model, str(ckpt))
            scan_model(model, tok, df_graph, df_retrieval, f"{arm}@{step}", args.n, args.alpha, rows)
            del model
            gc.collect(); torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
