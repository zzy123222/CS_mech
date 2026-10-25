from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.generate_error_monitor import build_check_prompt
from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook
from src.models.scoring import score_answer_pairs
from src.models.load import load_causal_lm
from src.utils.io import load_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed

MODULES = ["L20", "L22:mlp", "L25:attn"]
FORMAL = Path("posttraining/results/formal")
OUT = Path("results/p2_specificity")


def margin(bundle, prompt, pos, neg, hooks=None):
    z = score_answer_pairs(bundle, [prompt, prompt], [pos, neg], batch_size=2, hooks=hooks)
    return float(z[0] - z[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--models", nargs="+", default=[
        "base::Qwen/Qwen3-4B-Base", "ref::Qwen/Qwen3-4B",
        *[f"{a}_seed{s}@1000::posttraining/results/formal/{a}_seed{s}/step_1000"
          for a in ("general_sft", "correction_sft", "sham_correction") for s in (1, 2, 3)],
        *[f"{a}_seed1@200::posttraining/results/formal/{a}_seed1/step_0200"
          for a in ("general_sft", "correction_sft", "sham_correction")],
    ])
    args = ap.parse_args()
    set_seed(11)
    df = load_table("data/generated/p2_errors.parquet").sample(args.n, random_state=11)
    pos, neg = "CHECK", "CONTINUE"
    rows = []

    def prep_prompts(bundle):
        pre = []
        for _, row in df.iterrows():
            t = max(1, min(int(row.n_steps), int(row.error_step) + 1))
            pe = bundle.prepare_prompt(build_check_prompt(row, "error", t), enable_thinking=False)
            pc = bundle.prepare_prompt(build_check_prompt(row, "clean", t), enable_thinking=False)
            pre.append({"sid": row.sample_id, "pe": pe, "pc": pc,
                        "epos": len(bundle.tokenizer(pe, add_special_tokens=False)["input_ids"]) - 1,
                        "cpos": len(bundle.tokenizer(pc, add_special_tokens=False)["input_ids"]) - 1})
        return pre

    for spec_text in args.models:
        tag, path = spec_text.split("::")
        if path.startswith("Qwen/"):
            bundle = load_causal_lm(path, dtype="bfloat16", device_map="auto", chat=True)
        else:
            from peft import PeftModel
            bundle = load_causal_lm("Qwen/Qwen3-4B-Base", dtype="bfloat16",
                                    device_map="auto", chat=True)
            bundle.model = PeftModel.from_pretrained(bundle.model, path)
            bundle.model.eval()
        pre = prep_prompts(bundle)
        for r in pre:
            r["C_err"] = margin(bundle, r["pe"], pos, neg)
            r["C_clean"] = margin(bundle, r["pc"], pos, neg)
        clean_prompts = [r["pc"] for r in pre]
        for mod_text in MODULES:
            spec = ModuleSpec.parse(mod_text)
            module = resolve_module(bundle.model, spec)
            mv = estimate_mean_vector(bundle, clean_prompts, spec, max_samples=len(clean_prompts))
            for r in tqdm(pre, desc=f"{tag} {mod_text}", leave=False):
                he = (module, make_mean_ablation_hook(mv, r["epos"]))
                hc = (module, make_mean_ablation_hook(mv, r["cpos"]))
                rows.append({"model": tag, "module": mod_text, "sid": r["sid"],
                             "C_err": r["C_err"], "C_clean": r["C_clean"],
                             "C_err_abl": margin(bundle, r["pe"], pos, neg, [he]),
                             "C_clean_abl": margin(bundle, r["pc"], pos, neg, [hc])})
            pd.DataFrame(rows).to_parquet(OUT / "p03_endpoints.parquet")
        del bundle
        torch.cuda.empty_cache()
        print(f"[done] {tag}")

    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "p03_endpoints.parquet")
    summary = []
    for (model, mod), g in out.groupby(["model", "module"]):
        d_err = (g.C_err_abl - g.C_err).mean()
        d_clean = (g.C_clean_abl - g.C_clean).mean()
        rng = np.random.default_rng(777)
        de, dc = (g.C_err_abl - g.C_err).values, (g.C_clean_abl - g.C_clean).values
        idx = rng.integers(0, len(g), size=(10000, len(g)))
        stat = de[idx].mean(1) - dc[idx].mean(1)
        e, c = g.C_err.values, g.C_clean.values
        sp = np.sqrt((e.var(ddof=1) + c.var(ddof=1)) / 2)
        r = np.ones_like(e)
        auc = (pd.Series(np.concatenate([e, c])).rank().values[:len(e)].sum()
               - len(e) * (len(e) + 1) / 2) / (len(e) ** 2)
        summary.append({"model": model, "module": mod, "n": len(g),
                        "delta_err": d_err, "delta_clean": d_clean,
                        "delta_disc": d_err - d_clean,
                        "disc_ci_low": float(np.percentile(stat, 2.5)),
                        "disc_ci_high": float(np.percentile(stat, 97.5)),
                        "auc_pre": auc, "dprime_pre": (e.mean() - c.mean()) / (sp + 1e-9)})
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "p03_endpoints_summary.csv", index=False)
    print(sm.to_string(index=False))
    json.dump({"n": args.n, "models": [m.split("::")[0] for m in args.models],
               "date": "2026-10-14"}, open(OUT / "p03_endpoints_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
