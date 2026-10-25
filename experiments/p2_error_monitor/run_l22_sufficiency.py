from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.generate_error_monitor import build_check_prompt
from src.interventions.hooks import make_patch_hook
from src.interventions.patching import capture_module_vector
from src.models.scoring import score_answer_pairs
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed

MODULES = ["L22:mlp", "L22:attn", "L22"]
POSITIONS = ["terminal", "all"]
N_DONORS = 10


def margin(bundle, prompt, pos_ans, neg_ans, hooks=None):
    z = score_answer_pairs(bundle, [prompt, prompt], [pos_ans, neg_ans],
                           batch_size=2, hooks=hooks)
    return float(z[0] - z[1])


def rom_ci(num, den, n_boot=10000, seed=777):
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num), np.asarray(den)
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = num[idx].mean(1) / np.where(den[idx].mean(1) == 0, np.nan, den[idx].mean(1))
    stats = stats[~np.isnan(stats)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2_error_monitor.yaml")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="results/p2_l22_sufficiency")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 11))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    if len(df) > args.n:
        df = df.sample(args.n, random_state=seed).reset_index(drop=True)
    pos_ans = str(cfg.get("check_answer", "CHECK"))
    neg_ans = str(cfg.get("continue_answer", "CONTINUE"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = []
    for _, row in df.iterrows():
        t = max(1, min(int(row.n_steps), int(row.error_step) + 1))
        pe = bundle.prepare_prompt(build_check_prompt(row, "error", t),
                                   enable_thinking=mcfg.get("enable_thinking"))
        pc = bundle.prepare_prompt(build_check_prompt(row, "clean", t),
                                   enable_thinking=mcfg.get("enable_thinking"))
        pre.append({"sid": row.sample_id, "pe": pe, "pc": pc,
                    "epos": len(bundle.tokenizer(pe, add_special_tokens=False)["input_ids"]) - 1,
                    "cpos": len(bundle.tokenizer(pc, add_special_tokens=False)["input_ids"]) - 1})

    for r in pre:
        r["C_err"] = margin(bundle, r["pe"], pos_ans, neg_ans)
        r["C_clean"] = margin(bundle, r["pc"], pos_ans, neg_ans)

    rng = random.Random(seed + 1)
    rows = []

    def capture_seq(bundle, prompt, module):
        holder = {}

        def hook(_m, _inp, output):
            from src.hooks.core import first_tensor
            holder["v"] = first_tensor(output)[0].detach().float().cpu()
        h = module.register_forward_hook(hook)
        try:
            margin(bundle, prompt, "CHECK", "CONTINUE")
        finally:
            h.remove()
        return holder["v"]

    for mod_text in MODULES:
        spec = ModuleSpec.parse(mod_text)
        module = resolve_module(bundle.model, spec)
        for r in tqdm(pre, desc=f"{mod_text}", leave=False):
            v_err = capture_module_vector(bundle, r["pe"], spec, r["epos"])
            v_clean = capture_module_vector(bundle, r["pc"], spec, r["cpos"])
            seq_err = capture_seq(bundle, r["pe"], module)
            seq_clean = capture_seq(bundle, r["pc"], module)
            len_match = seq_err.shape[0] == seq_clean.shape[0]
            perm = torch.randperm(v_err.numel(),
                                  generator=torch.Generator().manual_seed(rng.randint(0, 2**31)))
            v_shuf = v_err.reshape(-1)[perm].reshape(v_err.shape)
            for pos_name in POSITIONS:
                if pos_name == "all" and not len_match:
                    continue
                if pos_name == "terminal":
                    src_dn, tp, cp, allp = v_err, r["epos"], r["cpos"], False
                    src_ns, src_shuf = v_clean, v_shuf
                else:
                    src_dn, tp, cp, allp = seq_err, -1, -1, True
                    src_ns, src_shuf = seq_clean, None
                hooks_d = [(module, make_patch_hook(src_dn, target_pos=cp, all_positions=allp))]
                c_dn = margin(bundle, r["pc"], pos_ans, neg_ans, hooks_d)
                hooks_n = [(module, make_patch_hook(src_ns, target_pos=tp, all_positions=allp))]
                c_ns = margin(bundle, r["pe"], pos_ans, neg_ans, hooks_n)
                c_shuf = (margin(bundle, r["pe"], pos_ans, neg_ans,
                                 [(module, make_patch_hook(src_shuf, target_pos=tp, all_positions=allp))])
                          if src_shuf is not None else np.nan)
                c_wrong = []
                used = {r["sid"]}
                attempts = 0
                while len(c_wrong) < N_DONORS and attempts < 20 * N_DONORS:
                    attempts += 1
                    w = pre[rng.randrange(len(pre))]
                    if w["sid"] in used:
                        continue
                    if pos_name == "terminal":
                        vw = capture_module_vector(bundle, w["pc"], spec, w["cpos"])
                    else:
                        vw = capture_seq(bundle, w["pc"], module)
                        if vw.shape[0] != seq_err.shape[0]:
                            continue
                    used.add(w["sid"])
                    c_wrong.append(margin(
                        bundle, r["pe"], pos_ans, neg_ans,
                        [(module, make_patch_hook(vw, target_pos=tp, all_positions=allp))]))
                rows.append({"module": mod_text, "position": pos_name,
                             "sid": r["sid"], "C_err": r["C_err"], "C_clean": r["C_clean"],
                             "C_clean_patched_err": c_dn, "C_err_patched_clean": c_ns,
                             "C_err_patched_shuffled": c_shuf,
                             **{f"C_err_patched_wrong{j}": v for j, v in enumerate(c_wrong)}})
        pd.DataFrame(rows).to_parquet(out_dir / "l22_sufficiency.parquet")
        print(f"[flush] {mod_text} done ({len(rows)} rows)")

    out = pd.DataFrame(rows)
    out.to_parquet(out_dir / "l22_sufficiency.parquet")
    summary = []
    for (mod, pos), g in out.groupby(["module", "position"]):
        gap = (g.C_err - g.C_clean).mean()
        denoise_frac = (g.C_clean_patched_err - g.C_clean).mean() / gap if gap else np.nan
        noise_frac = (g.C_err_patched_clean - g.C_err).mean() / -gap if gap else np.nan
        wrong_cols = [c for c in g.columns if c.startswith("C_err_patched_wrong")]
        wrong_frac = (g[wrong_cols].mean(axis=1) - g.C_err).mean() / -gap if gap else np.nan
        shuf_frac = (g.C_err_patched_shuffled - g.C_err).mean() / -gap if gap else np.nan
        d_lo, d_hi = rom_ci(g.C_clean_patched_err - g.C_clean, g.C_err - g.C_clean)
        n_lo, n_hi = rom_ci(g.C_err_patched_clean - g.C_err, g.C_clean - g.C_err)
        w_lo, w_hi = rom_ci(g[wrong_cols].mean(axis=1) - g.C_err, g.C_clean - g.C_err)
        summary.append({
            "module": mod, "position": pos, "n": len(g),
            "mean_C_err": g.C_err.mean(), "mean_C_clean": g.C_clean.mean(), "gap": gap,
            "denoise_frac": denoise_frac, "denoise_ci": f"[{d_lo:.3f},{d_hi:.3f}]",
            "noise_frac": noise_frac, "noise_ci": f"[{n_lo:.3f},{n_hi:.3f}]",
            "wrong10_frac": wrong_frac, "wrong_ci": f"[{w_lo:.3f},{w_hi:.3f}]",
            "shuffled_frac": shuf_frac,
            "denoise_ge_0.3_and_gt_wrong": bool(
                denoise_frac >= 0.3 and d_lo > wrong_frac),
        })
    sm = pd.DataFrame(summary)
    sm.to_csv(out_dir / "l22_sufficiency_summary.csv", index=False)
    print(sm.to_string(index=False))
    json.dump({"config": cfg["model_config"], "n": len(pre), "seed": seed,
               "n_donors": N_DONORS, "date": "2026-10-14"},
              open(out_dir / "run_metadata.json", "w"), indent=1)


if __name__ == "__main__":
    main()
