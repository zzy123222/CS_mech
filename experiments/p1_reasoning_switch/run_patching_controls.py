from __future__ import annotations

import argparse, random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.interventions.patching import capture_module_vector, margin_with_hook
from src.interventions.hooks import make_patch_hook
from src.utils.experiment import load_bundle_from_model_config
from src.utils.io import load_yaml, load_table, parse_int_list, save_table
from src.utils.landmarks import landmark_token_position
from src.utils.module_access import ModuleSpec, get_decoder_layers, resolve_module
from src.utils.seed import set_seed


def choose_donor(rng: random.Random, ronly, target_gid: str, target_rule_answer: str,
                 kind: str, pool_by_answer):
    if kind == "wrong_donor":
        cands = [r for r in ronly.itertuples() if r.group_id != target_gid]
        return rng.choice(cands)
    if kind == "same_answer_donor":
        cands = [r for r in pool_by_answer.get(target_rule_answer, []) if r.group_id != target_gid]
        if not cands:
            cands = [r for r in ronly.itertuples() if r.group_id != target_gid]
        return rng.choice(cands)
    raise ValueError(f"unknown donor kind {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", nargs="*", default=["all"])
    ap.add_argument("--kind", choices=["resid", "attn", "mlp"], default="resid")
    ap.add_argument("--controls", nargs="+", default=["wrong_donor", "same_answer_donor"],
                    choices=["wrong_donor", "same_answer_donor"])
    ap.add_argument("--positions", nargs="*", default=["last"])
    ap.add_argument("--max_pairs", type=int, default=64)
    ap.add_argument("--min_denom", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    set_seed(seed)
    bundle, mcfg = load_bundle_from_model_config(cfg["model_config"])
    df = load_table(cfg["dataset"])
    p_cfg = cfg.get("patching", {})
    clean_cond, corrupt_cond = p_cfg.get("clean_condition", "R-only"), p_cfg.get("corrupt_condition", "R+S-conflict")
    c = df[df.condition == clean_cond].set_index("group_id")
    x = df[df.condition == corrupt_cond].set_index("group_id")
    ids = list(c.index.intersection(x.index))
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[: args.max_pairs]
    ronly = df[df.condition == clean_cond]
    pool_by_answer = {}
    for r in ronly.itertuples():
        pool_by_answer.setdefault(str(r.rule_answer), []).append(r)
    nl = len(get_decoder_layers(bundle.model))
    layers = list(range(nl)) if args.layers == ["all"] else [i for t in args.layers for i in parse_int_list(t)]
    rows = []
    for gid in tqdm(ids, desc="donor-control pairs"):
        corrupt = x.loc[gid]
        clean = c.loc[gid]
        corrupt_prompt = bundle.prepare_prompt(corrupt.prompt, enable_thinking=mcfg.get("enable_thinking"))
        clean_prompt = bundle.prepare_prompt(clean.prompt, enable_thinking=mcfg.get("enable_thinking"))
        r, s = str(corrupt.rule_answer), str(corrupt.shortcut_answer)
        donors = {}
        for kind in args.controls:
            d = choose_donor(rng, ronly, gid, str(corrupt.rule_answer), kind, pool_by_answer)
            donors[kind] = (bundle.prepare_prompt(d.prompt, enable_thinking=mcfg.get("enable_thinking")), d.prompt)
        clean_margin = margin_with_hook(bundle, clean_prompt, r, s)
        corrupt_margin = margin_with_hook(bundle, corrupt_prompt, r, s)
        denom = clean_margin - corrupt_margin
        eps = 1e-8 if denom >= 0 else -1e-8
        for pname in args.positions:
            tp = landmark_token_position(bundle.tokenizer, corrupt_prompt, corrupt.prompt, pname)
            sp_matched = landmark_token_position(bundle.tokenizer, clean_prompt, clean.prompt, pname)
            for lid in layers:
                spec = ModuleSpec(lid, args.kind)
                try:
                    module = resolve_module(bundle.model, spec)
                    src_matched = capture_module_vector(bundle, clean_prompt, spec, sp_matched)
                    matched = margin_with_hook(bundle, corrupt_prompt, r, s,
                                               hooks=[(module, make_patch_hook(src_matched, target_pos=tp))])
                    rows.append({"group_id": gid, "layer": lid, "kind": args.kind, "position": pname,
                                 "clean_margin": clean_margin, "corrupt_margin": corrupt_margin, "denom": denom,
                                 "recovery": (matched - corrupt_margin) / (denom + eps),
                                 "control": "matched_donor"})
                    for kind, (dprompt, draw) in donors.items():
                        sp = landmark_token_position(bundle.tokenizer, dprompt, draw, pname)
                        src = capture_module_vector(bundle, dprompt, spec, sp)
                        val = margin_with_hook(bundle, corrupt_prompt, r, s,
                                               hooks=[(module, make_patch_hook(src, target_pos=tp))])
                        rows.append({"group_id": gid, "layer": lid, "kind": args.kind, "position": pname,
                                     "clean_margin": clean_margin, "corrupt_margin": corrupt_margin, "denom": denom,
                                     "recovery": (val - corrupt_margin) / (denom + eps),
                                     "control": kind})
                except Exception as e:
                    rows.append({"group_id": gid, "layer": lid, "kind": args.kind, "position": pname,
                                 "control": "error", "error": repr(e)})
    out_dir = Path(cfg.get("output_dir", "results/p1_reasoning_switch")) / "patching"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    save_table(out, out_dir / f"donor_controls_{args.kind}.parquet")
    ok = out[(out.control != "error") & (out.denom.abs() >= args.min_denom)]
    sm = ok.groupby(["control", "layer", "position"]).recovery.agg(["mean", "sem", "count"]).reset_index()
    save_table(sm, out_dir / f"donor_controls_{args.kind}_summary.csv")
    if len(sm):
        best = sm[sm.control == "matched_donor"].sort_values("mean").tail(1)
        if len(best):
            L = int(best.layer.iloc[0]); posname = best.position.iloc[0]
            cmp = sm[(sm.layer == L) & (sm.position == posname)]
            print(f"\npeak matched_donor recovery at L{L}/{posname}:")
            print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
