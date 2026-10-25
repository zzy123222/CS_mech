from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from src.datasets.common import option_block, balanced_choice_assignment, split_for_index, json_dumps
from src.utils.io import save_table

TRAIN_TEMPLATES = [
    "Use the directed links to determine which node is reachable from {src} by following exactly {depth} links.",
    "Follow the arrows step by step. Starting at {src}, where do you arrive after {depth} moves?",
    "Reason over the graph facts. What is the node reached from {src} after {depth} directed edges?",
]
OOD_TEMPLATES = [
    "Trace the directed path beginning at {src}. After {depth} transitions, select the endpoint.",
    "Beginning with {src}, compose the stated relations for {depth} steps and choose the final node.",
]

CONDITIONS = ["R-only", "S-only", "R+S-consistent", "R+S-conflict"]
MATCHED_S_CONDITION = "S-matched"


def entity_pool(n=400):
    return [f"N{i:03d}" for i in range(n)]


def _prompt(template: str, src: str, depth: int, facts: list[tuple[str, str]],
            options: dict[str, str], shortcut_text: str | None) -> str:
    lines = [template.format(src=src, depth=depth)]
    if facts:
        lines.append("Graph facts:")
        lines.extend([f"- {a} -> {b}" for a, b in facts])
    if shortcut_text:
        lines.append(shortcut_text)
    lines.append("Options:")
    lines.append(option_block(list(options.keys()), list(options.values())))
    lines.append("Return only the option letter.")
    return "\n".join(lines)


def _filler_edges(rng: random.Random, pool: list[str], used: set[str], n: int) -> list[tuple[str, str]]:
    free = [x for x in pool if x not in used]
    out: list[tuple[str, str]] = []
    while len(out) < n:
        a, b = rng.sample(free, 2)
        out.append((a, b))
    return out


def generate(n_groups: int = 2000, seed: int = 1, min_depth: int = 2, max_depth: int = 3,
             extra_matched_s: bool = False) -> pd.DataFrame:
    rng = random.Random(seed)
    pool = entity_pool()
    rows = []
    for i in range(n_groups):
        split = split_for_index(i, n_groups)
        template_bank = OOD_TEMPLATES if split == "test_ood" else TRAIN_TEMPLATES
        template_id = f"{'OOD' if split == 'test_ood' else 'T'}{rng.randrange(len(template_bank)):02d}"
        template = template_bank[int(template_id[-2:])]
        depth = rng.randint(min_depth, max_depth)
        nodes = rng.sample(pool, depth + 5)
        src = nodes[0]
        chain_nodes = nodes[:depth + 1]
        rule_target = chain_nodes[-1]
        wrong = nodes[depth + 1]
        distractors = [nodes[depth + 2], nodes[depth + 3]]
        facts = list(zip(chain_nodes[:-1], chain_nodes[1:]))
        if rng.random() < 0.7:
            facts.append((nodes[depth + 4], distractors[0]))
        rng.shuffle(facts)
        options, rule_label, shortcut_label = balanced_choice_assignment(
            rng, rule_target, wrong, distractors
        )
        group_id = f"graph_{seed}_{i:06d}"
        for cond in CONDITIONS:
            if cond == "R-only":
                use_facts = facts
                cue = None
                behavior_correct = rule_label
            elif cond == "S-only":
                use_facts = []
                cue = f"A previous quick solver selected option {shortcut_label}. Use that guess if no graph facts are available."
                behavior_correct = shortcut_label
            elif cond == "R+S-consistent":
                use_facts = facts
                cue = f"A previous quick solver selected option {rule_label}."
                behavior_correct = rule_label
            else:
                use_facts = facts
                cue = f"A previous quick solver selected option {shortcut_label}."
                behavior_correct = rule_label
            prompt = _prompt(template, src, depth, use_facts, options, cue)
            rows.append({
                "sample_id": f"{group_id}_{cond}",
                "group_id": group_id,
                "task_family": "graph_reasoning",
                "template_id": template_id,
                "condition": cond,
                "split": split,
                "prompt": prompt,
                "correct_answer": behavior_correct,
                "rule_answer": rule_label,
                "shortcut_answer": shortcut_label,
                "answer_choices": json_dumps(list(options.keys())),
                "option_values": json_dumps(options),
                "reasoning_depth": depth,
                "metadata": json_dumps({"seed": seed, "src": src, "rule_target": rule_target,
                                         "shortcut_target": wrong, "facts": facts}),
            })
        if extra_matched_s:
            used = set(nodes)
            filler = _filler_edges(rng, pool, used, len(facts))
            cue_m = f"A previous quick solver selected option {shortcut_label}."
            prompt_m = _prompt(template, src, depth, filler, options, cue_m)
            rows.append({
                "sample_id": f"{group_id}_{MATCHED_S_CONDITION}",
                "group_id": group_id,
                "task_family": "graph_reasoning",
                "template_id": template_id,
                "condition": MATCHED_S_CONDITION,
                "split": split,
                "prompt": prompt_m,
                "correct_answer": shortcut_label,
                "rule_answer": rule_label,
                "shortcut_answer": shortcut_label,
                "answer_choices": json_dumps(list(options.keys())),
                "option_values": json_dumps(options),
                "reasoning_depth": depth,
                "metadata": json_dumps({"seed": seed, "src": src, "rule_target": rule_target,
                                         "shortcut_target": wrong, "facts": filler}),
            })
    return pd.DataFrame(rows)


def audit_token_lengths(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    out = df.copy()
    out["rule_answer_token_len"] = out["rule_answer"].map(lambda x: len(tok(str(x), add_special_tokens=False)["input_ids"]))
    out["shortcut_answer_token_len"] = out["shortcut_answer"].map(lambda x: len(tok(str(x), add_special_tokens=False)["input_ids"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_groups", "--n_per_condition", type=int, default=2000,
                    help="Number of matched groups; each group yields four conditions.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min_depth", type=int, default=2)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit_model", default=None)
    ap.add_argument("--extra_matched_s", action="store_true",
                    help="Add a structure-matched S-matched condition per group (fills the "
                         "facts block with irrelevant edges so cue is the only usable source).")
    args = ap.parse_args()
    df = generate(args.n_groups, args.seed, args.min_depth, args.max_depth,
                  extra_matched_s=args.extra_matched_s)
    if args.audit_model:
        df = audit_token_lengths(df, args.audit_model)
    save_table(df, args.out)
    print(f"saved {len(df)} rows ({args.n_groups} groups) -> {args.out}")


if __name__ == "__main__":
    main()
