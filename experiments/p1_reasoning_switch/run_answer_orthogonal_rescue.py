from __future__ import annotations

import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.interventions.patching import capture_module_vector, margin_with_hook
from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook, make_patch_hook
from src.models.scoring import single_token_id
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, save_table
from src.utils.landmarks import landmark_token_position
from src.utils.module_access import ModuleSpec, resolve_module, get_decoder_layers
from src.hooks.core import first_tensor
from src.stats.bootstrap import bootstrap_mean
from src.utils.seed import set_seed


def orthonormalize(cols: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(cols.float())
    return q


def unembedding_basis(bundle, letters=("A", "B", "C", "D")):
    w = bundle.model.get_output_embeddings().weight.detach().float()
    rows = []
    for lt in letters:
        tid = single_token_id(bundle.tokenizer, lt)
        rows.append(w[tid])
    return orthonormalize(torch.stack(rows, dim=1))


def gradient_basis(bundle, mcfg, spec, items, k=4):
    module = resolve_module(bundle.model, spec)
    device = next(bundle.model.parameters()).device
    grads = []
    for raw, rid, sid in items:
        p = bundle.prepare_prompt(raw, enable_thinking=mcfg.get("enable_thinking"))
        holder = {}
        def hook(_m, _i, output):
            h = first_tensor(output)
            h.retain_grad()
            holder["h"] = h
        handle = module.register_forward_hook(hook)
        try:
            enc = bundle.tokenizer(p, return_tensors="pt", add_special_tokens=False)
            enc = {kk: v.to(device) for kk, v in enc.items()}
            out = bundle.model(**enc, use_cache=False)
            metric = out.logits[0, -1, rid] - out.logits[0, -1, sid]
            bundle.model.zero_grad(set_to_none=True)
            metric.backward()
        finally:
            handle.remove()
        pp = holder["h"].shape[1] - 1
        grads.append(holder["h"].grad[0, pp].detach().float().cpu())
        bundle.model.zero_grad(set_to_none=True)
    G = torch.stack(grads)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    return Vh[:k, :].T.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--module", default="L24:attn")
    ap.add_argument("--clean_condition", default="R+S-consistent")
    ap.add_argument("--corrupt_condition", default="R+S-conflict")
    ap.add_argument("--basis", choices=["unembedding", "gradient", "both"], default="both")
    ap.add_argument("--k_grad", type=int, default=4)
    ap.add_argument("--n_grad", type=int, default=32)
    ap.add_argument("--max_pairs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    spec = ModuleSpec.parse(args.module)
    module = resolve_module(bundle.model, spec)

    letters = ("A", "B", "C", "D")
    r_ids = [single_token_id(bundle.tokenizer, l) for l in letters]

    cl = df[df.condition == args.clean_condition]
    co = df[df.condition == args.corrupt_condition].set_index("group_id")
    cl_by_group = cl.set_index("group_id")
    ids = list(cl_by_group.index.intersection(co.index))
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[: args.max_pairs]
    clean_rows = cl
    pool_by_letter = {}
    for r_ in clean_rows.itertuples():
        pool_by_letter.setdefault(str(r_.rule_answer), []).append(r_)

    bases = {}
    if args.basis in ("unembedding", "both"):
        bases["unemb"] = unembedding_basis(bundle, letters).float().cpu()
    if args.basis in ("gradient", "both"):
        items = [(t.prompt, single_token_id(bundle.tokenizer, str(t.rule_answer)),
                  single_token_id(bundle.tokenizer, str(t.shortcut_answer)))
                 for t in clean_rows.head(args.n_grad).itertuples()]
        bases["grad"] = gradient_basis(bundle, mcfg, spec, items, k=args.k_grad)

    ctl_ids = ids[: min(len(ids), 96)]
    ctl_prompts = [bundle.prepare_prompt(cl_by_group.loc[g].prompt, enable_thinking=mcfg.get("enable_thinking")) for g in ctl_ids]
    mean_vec = estimate_mean_vector(bundle, ctl_prompts, spec, max_samples=len(ctl_prompts))

    rows = []
    for gid in tqdm(ids, desc=f"answer-ortho {args.module}"):
        c_row, x_row = cl_by_group.loc[gid], co.loc[gid]
        cp = bundle.prepare_prompt(c_row.prompt, enable_thinking=mcfg.get("enable_thinking"))
        xp = bundle.prepare_prompt(x_row.prompt, enable_thinking=mcfg.get("enable_thinking"))
        r, s = str(x_row.rule_answer), str(x_row.shortcut_answer)
        xpos = landmark_token_position(bundle.tokenizer, xp, x_row.prompt, "last")

        donors = {"matched": capture_module_vector(bundle, cp, spec,
                                                   landmark_token_position(bundle.tokenizer, cp, c_row.prompt, "last"))}
        cands = [t for t in pool_by_letter.get(str(x_row.rule_answer), []) if t.group_id != gid]
        d1 = rng.choice(cands) if cands else rng.choice([t for t in clean_rows.itertuples() if t.group_id != gid])
        cands2 = [t for t in clean_rows.itertuples() if t.group_id != gid]
        d2 = rng.choice(cands2)
        for nm, d in (("same_answer", d1), ("wrong", d2)):
            dp = bundle.prepare_prompt(d.prompt, enable_thinking=mcfg.get("enable_thinking"))
            donors[nm] = capture_module_vector(bundle, dp, spec,
                                               landmark_token_position(bundle.tokenizer, dp, d.prompt, "last"))

        m_base = margin_with_hook(bundle, xp, r, s)
        ko_hook = (module, make_mean_ablation_hook(mean_vec, xpos))
        m_ko = margin_with_hook(bundle, xp, r, s, [ko_hook])
        denom = m_base - m_ko
        eps = 1e-8 if denom >= 0 else -1e-8
        rec = {"group_id": gid, "m_base": m_base, "m_ko": m_ko, "denom": denom}

        for bname, Q in bases.items():
            for dname, a in donors.items():
                proj = Q @ (Q.T @ a)
                a_ans, a_perp = proj, a - proj
                for cname, comp in (("full", a), ("answer", a_ans), ("perp", a_perp)):
                    m = margin_with_hook(bundle, xp, r, s,
                                         [ko_hook, (module, make_patch_hook(comp, target_pos=xpos))])
                    rec[f"{bname}/{dname}/{cname}"] = (m - m_ko) / (denom + eps)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_dir = Path(cfg.get("output_dir", "results/p1_reasoning_switch")) / "rescue"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(out, out_dir / f"answer_ortho_{args.module.replace(':', '_')}.parquet")
    ok = out[out.denom.abs() >= 0.25]
    summary = []
    for col in [c for c in ok.columns if "/" in c]:
        ci = bootstrap_mean(ok[col].values, n_boot=2000, seed=seed)
        bname, dname, cname = col.split("/")
        summary.append({"basis": bname, "donor": dname, "component": cname,
                        "mean": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": len(ok)})
    sm = pd.DataFrame(summary)
    save_table(sm, out_dir / f"answer_ortho_{args.module.replace(':', '_')}_summary.csv")
    print(sm.pivot_table(index=["basis", "donor"], columns="component", values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
