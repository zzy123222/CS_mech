from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Iterable

CHOICE_LABELS = ["A", "B", "C", "D"]


def option_block(labels: list[str], values: list[str]) -> str:
    return "\n".join(f"{l}. {v}" for l, v in zip(labels, values))


def balanced_choice_assignment(rng: random.Random, correct_value: str, shortcut_value: str,
                               distractors: list[str]):
    values = [correct_value, shortcut_value] + distractors
    uniq = []
    for x in values:
        if x not in uniq:
            uniq.append(x)
    if len(uniq) != 4:
        raise ValueError(f"Need exactly four unique option values, got {uniq}")
    rng.shuffle(uniq)
    mapping = dict(zip(CHOICE_LABELS, uniq))
    inv = {v: k for k, v in mapping.items()}
    return mapping, inv[correct_value], inv[shortcut_value]


def split_for_index(i: int, n: int) -> str:
    frac = (i + 0.5) / max(n, 1)
    if frac < 0.70:
        return "train"
    if frac < 0.85:
        return "val"
    return "test_ood"


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def parse_choices(x) -> list[str]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            if isinstance(y, list):
                return [str(z) for z in y]
        except Exception:
            pass
        return [z.strip() for z in x.split("|")]
    return list(x)
