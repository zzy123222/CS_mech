from __future__ import annotations
import sys, json, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
import torch

from src.interventions.ablation import estimate_mean_vector
from src.interventions.hooks import make_mean_ablation_hook
from src.models.scoring import score_answer_pairs
from src.utils.io import load_table
from src.utils.module_access import ModuleSpec, resolve_module
from src.utils.seed import set_seed


def load_merged_model(base_id, adapter_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def paired_bootstrap_ci(x, n_boot=10000, seed=1234):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class Panel:

    def __init__(self, model, tok):
        self.model = model
        self.tokenizer = tok

    def margin_batched(self, prompts, answers_a, answers_b, batch_size=8):
        prompts_flat, answers_flat = [], []
        for p, a, b in zip(prompts, answers_a, answers_b):
            prompts_flat += [p, p]
            answers_flat += [str(a), str(b)]
        scores = score_answer_pairs(self, prompts_flat, answers_flat, batch_size=batch_size)
        scores = np.asarray(scores, dtype=float)
        return scores[0::2] - scores[1::2]

    def margin_hooked(self, prompt, ans_a, ans_b, hook):
        sc = score_answer_pairs(self, [prompt, prompt], [str(ans_a), str(ans_b)],
                                batch_size=2, hooks=[hook])
        return float(sc[0] - sc[1])

    def last_pos(self, prompt):
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]) - 1


def measure_panel(model, tok, df_p2, df_graph, df_retrieval, n=64):
    panel = Panel(model, tok)
    from src.datasets.generate_error_monitor import build_check_prompt
    res, arrays = {}, {}
    CHECK, CONT = "CHECK", "CONTINUE"

    p2_rows = df_p2.head(n)
    p2_prompts, p2_clean_prompts = [], []
    for _, row in p2_rows.iterrows():
        t = max(1, min(int(row.n_steps), int(row.error_step) + 1))
        raw_err = build_check_prompt(row, "error", t)
        p2_prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": raw_err}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False))
        raw_cln = build_check_prompt(row, "clean", t)
        p2_clean_prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": raw_cln}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False))

    base_m = panel.margin_batched(p2_prompts, [CHECK]*len(p2_prompts), [CONT]*len(p2_prompts))
    cln_m = panel.margin_batched(p2_clean_prompts, [CHECK]*len(p2_clean_prompts), [CONT]*len(p2_clean_prompts))
    arrays["p2_error_margin"] = base_m
    arrays["p2_clean_margin"] = cln_m
    res["gate_p2_error_acc"] = float((base_m > 0).mean())
    res["gate_p2_clean_acc"] = float((cln_m < 0).mean())

    specs = {"L20": ModuleSpec(20, "resid"), "L22mlp": ModuleSpec(22, "mlp"),
             "L24attn": ModuleSpec(24, "attn"), "L25attn": ModuleSpec(25, "attn")}
    mods = {k: resolve_module(model, s) for k, s in specs.items()}
    means = {k: estimate_mean_vector(panel, p2_clean_prompts, s, max_samples=min(n, 16))
             for k, s in specs.items()}

    for key, tag in [("L20", "monitor"), ("L22mlp", "controller"), ("L25attn", "opponent")]:
        deltas = np.empty(len(p2_prompts))
        for i, prompt in enumerate(p2_prompts):
            hook = (mods[key], make_mean_ablation_hook(means[key], panel.last_pos(prompt)))
            deltas[i] = panel.margin_hooked(prompt, CHECK, CONT, hook) - base_m[i]
        arrays[f"{tag}_per_sample"] = deltas
        res[f"{tag}_d_{key}"] = float(deltas.mean())
        lo, hi = paired_bootstrap_ci(deltas)
        res[f"{tag}_d_{key}_ci_low"], res[f"{tag}_d_{key}_ci_high"] = lo, hi

    def task_damage(df_task, task_tag, n_task):
        clean = df_task[df_task.condition == "R-only"].head(n_task)
        conflict = df_task[df_task.condition == "R+S-conflict"].head(n_task)
        clean_prompts = [tok.apply_chat_template(
            [{"role": "user", "content": r.prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        ) for _, r in clean.iterrows()]
        mean_vec = estimate_mean_vector(panel, clean_prompts, specs["L24attn"],
                                        max_samples=min(n_task, 16))
        conf_prompts, ra, sa = [], [], []
        for _, r in conflict.iterrows():
            conf_prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": r.prompt}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
            ra.append(r.rule_answer); sa.append(r.shortcut_answer)
        base = panel.margin_batched(conf_prompts, ra, sa)
        ronly = panel.margin_batched(clean_prompts,
                                     [r.rule_answer for _, r in clean.iterrows()],
                                     [r.shortcut_answer for _, r in clean.iterrows()])
        dmg = np.empty(len(conf_prompts))
        for i, prompt in enumerate(conf_prompts):
            hook = (mods["L24attn"], make_mean_ablation_hook(mean_vec, panel.last_pos(prompt)))
            dmg[i] = base[i] - panel.margin_hooked(prompt, ra[i], sa[i], hook)
        arrays[f"integrator_{task_tag}_per_sample"] = dmg
        arrays[f"{task_tag}_conflict_margin"] = base
        return dmg, float((ronly > 0).mean())

    g_dmg, res["gate_graph_ronly_acc"] = task_damage(df_graph, "graph", n)
    r_dmg, res["gate_retrieval_ronly_acc"] = task_damage(df_retrieval, "retrieval", n)
    res["integrator_d_L24attn"] = float((g_dmg.mean() + r_dmg.mean()) / 2)
    res["integrator_d_L24attn_graph"] = float(g_dmg.mean())
    res["integrator_d_L24attn_retrieval"] = float(r_dmg.mean())
    lo, hi = paired_bootstrap_ci(np.concatenate([g_dmg, r_dmg]))
    res["integrator_d_L24attn_ci_low"], res["integrator_d_L24attn_ci_high"] = lo, hi

    res["gate_graph_pass"] = bool(res["gate_graph_ronly_acc"] >= 0.65)
    res["gate_retrieval_pass"] = bool(res["gate_retrieval_ronly_acc"] >= 0.65)
    res["gate_p2_pass"] = bool(res["gate_p2_error_acc"] >= 0.65)
    res["integrator_interpretable"] = bool(res["gate_graph_pass"] and res["gate_retrieval_pass"])
    res["p2_indices_interpretable"] = bool(res["gate_p2_pass"])

    return res, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_dir", default=None, help="e.g. posttraining/results/general_sft_seed1")
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--direct_model", default=None,
                    help="scan a bare model (no adapter), e.g. Qwen/Qwen3-4B -> reference_metrics.json")
    ap.add_argument("--output", default=None)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    set_seed(1)
    df_p2 = load_table("data/generated/p2_errors.parquet")
    battery = load_table("data/generated/p3_task_battery.parquet")
    df_graph = battery[battery.task_family == "graph_reasoning"]
    df_retrieval = battery[battery.task_family == "retrieval"]

    if args.direct_model:
        print(f"=== Direct-model reference scan: {args.direct_model} ===", flush=True)
        model, tok = load_merged_model(args.direct_model, None)
        res, arrays = measure_panel(model, tok, df_p2, df_graph, df_retrieval, n=args.n)
        res["model"] = args.direct_model
        res["n"] = args.n
        out_json = args.output or str(ROOT / "posttraining/results/reference_metrics.json")
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
        np.savez(Path(out_json).with_suffix(".npz"), **arrays)
        print(json.dumps(res, indent=2))
        print(f"saved: {out_json}")
        return

    arm_dir = Path(args.arm_dir)
    checkpoints = sorted(arm_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))

    all_results = []
    npz_store = {}
    for ckpt_dir in checkpoints:
        step = int(ckpt_dir.name.split("_")[1])
        print(f"\n─── Checkpoint step={step} ({ckpt_dir}) ───", flush=True)

        model, tok = load_merged_model(args.base_model, str(ckpt_dir))
        res, arrays = measure_panel(model, tok, df_p2, df_graph, df_retrieval, n=args.n)
        res["step"] = step
        res["checkpoint"] = str(ckpt_dir)
        res["arm"] = arm_dir.name
        all_results.append(res)
        for k, v in arrays.items():
            npz_store[f"step{step:04d}_{k}"] = v

        print(f"  Monitor dE(L20):     {res['monitor_d_L20']:+.4f} "
              f"[{res['monitor_d_L20_ci_low']:+.4f}, {res['monitor_d_L20_ci_high']:+.4f}]")
        print(f"  Controller dC(L22):  {res['controller_d_L22mlp']:+.4f} "
              f"[{res['controller_d_L22mlp_ci_low']:+.4f}, {res['controller_d_L22mlp_ci_high']:+.4f}]")
        print(f"  Integrator(L24):     {res['integrator_d_L24attn']:+.4f} "
              f"[{res['integrator_d_L24attn_ci_low']:+.4f}, {res['integrator_d_L24attn_ci_high']:+.4f}]"
              f"  graph {res['integrator_d_L24attn_graph']:+.4f} / retr {res['integrator_d_L24attn_retrieval']:+.4f}")
        print(f"  Opponent dC(L25):    {res['opponent_d_L25attn']:+.4f} "
              f"[{res['opponent_d_L25attn_ci_low']:+.4f}, {res['opponent_d_L25attn_ci_high']:+.4f}]")
        print(f"  Gates: graph {res['gate_graph_ronly_acc']:.2f} retr {res['gate_retrieval_ronly_acc']:.2f} "
              f"p2err {res['gate_p2_error_acc']:.2f} p2cln {res['gate_p2_clean_acc']:.2f} "
              f"| integrator interpretable: {res['integrator_interpretable']}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    out = args.output or str(arm_dir / "mechanism_panel.csv")
    df.to_csv(out, index=False)
    np.savez(arm_dir / "mechanism_panel_persample.npz", **npz_store)
    print(f"\nSaved: {out}")

    cols = ["step", "monitor_d_L20", "controller_d_L22mlp",
            "integrator_d_L24attn", "opponent_d_L25attn",
            "gate_graph_ronly_acc", "gate_retrieval_ronly_acc", "gate_p2_error_acc"]
    print("\n" + "=" * 70)
    print("DEVELOPMENTAL SUMMARY")
    print("=" * 70)
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
