from __future__ import annotations

import numpy as np

from src.models.scoring import score_answer_pairs

GENERIC_PAIRS = [
    ("The capital of France is", " Paris"),
    ("2 + 2 =", " 4"),
    ("Complete the sequence: A B C", " D"),
    ("The opposite of hot is", " cold"),
    ("The sky on a clear day is", " blue"),
    ("Count: one two three", " four"),
    ("The first day of the week is", " Monday"),
    ("The reverse of up is", " down"),
    ("5 times 3 equals", " 15"),
    ("Paris is a city in", " France"),
    ("Red, green, and blue are all", " colors"),
    ("10 minus 4 equals", " 6"),
    ("A very common greeting is", " Hello"),
    ("The letter after A is", " B"),
    ("Copy this word exactly: tree. tree", " tree"),
    ("An apple is a kind of", " fruit"),
]


def _prepared(bundle, mcfg, pairs):
    out = []
    for raw, ans in pairs:
        p = bundle.prepare_prompt(raw, enable_thinking=mcfg.get("enable_thinking"))
        n_tok = max(1, len(bundle.tokenizer(ans, add_special_tokens=False)["input_ids"]))
        out.append((p, ans, n_tok))
    return out


def build_extended_battery(n=120, seed=0):
    import random
    rng = random.Random(seed)
    items = list(GENERIC_PAIRS)

    def add(prompt, answer):
        items.append((prompt, answer))

    for _ in range(24):
        a, b = rng.randint(11, 89), rng.randint(2, 9)
        kind = rng.choice(["add", "sub", "mul", "bigger"])
        if kind == "add":
            add(f"{a} + {b} =", f" {a + b}")
        elif kind == "sub":
            add(f"{a} - {b} =", f" {a - b}")
        elif kind == "mul":
            add(f"{a} times {b} equals", f" {a * b}")
        else:
            c = rng.randint(2, a - 1)
            add(f"Which number is bigger: {c} or {a}?", f" {a}")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for _ in range(12):
        i = rng.randint(0, 20)
        add(f"Complete the sequence: {letters[i]} {letters[i+1]} {letters[i+2]}",
            f" {letters[i+3]}")
        s, k = rng.randint(1, 9), rng.randint(2, 5)
        seq = " ".join(str(s + k * j) for j in range(3))
        add(f"Continue the number sequence: {seq}", f" {s + k * 3}")
    for w in ["tree", "river", "stone", "cloud", "paper", "glass", "chair", "bread",
              "night", "field", "brush", "steel"]:
        add(f"Copy this word exactly: {w}. {w}", f" {w}")
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    for i in range(11):
        add(f"The month after {months[i]} is", f" {months[i + 1]}")
    for a, b in [("cat", "dog"), ("sun", "moon"), ("fast", "slow"), ("big", "small"),
                 ("hot", "cold"), ("up", "down"), ("day", "night"), ("open", "close"),
                 ("long", "short"), ("old", "new"), ("wet", "dry"), ("rich", "poor")]:
        add(f"The opposite of {a} is", f" {b}")
    for c, cap in [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
                   ("Spain", "Madrid"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
                   ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("China", "Beijing"),
                   ("India", "New Delhi"), ("Russia", "Moscow"), ("Peru", "Lima")]:
        add(f"The capital of {c} is", f" {cap}")
    for _ in range(10):
        m = rng.randint(2, 12)
        add(f"Is {m} a number? Answer yes or no.", " yes")
    rng.shuffle(items)
    return items[:max(n, 100)] if n >= 100 else items[:n]


EXTENDED_PAIRS = build_extended_battery()


def generic_nll(bundle, mcfg, hooks=None, pairs=None) -> float:
    items = _prepared(bundle, mcfg, pairs or GENERIC_PAIRS)
    scores = score_answer_pairs(bundle, [p for p, _, _ in items], [a for _, a, _ in items],
                                batch_size=4, hooks=hooks)
    return float(np.mean([-s / n for s, (_, _, n) in zip(scores, items)]))


def generic_nll_with_hook_factory(bundle, mcfg, hook_factory, pairs=None) -> float:
    total, count = 0.0, 0
    for p, ans, n_tok in _prepared(bundle, mcfg, pairs or GENERIC_PAIRS):
        pos = len(bundle.tokenizer(p, add_special_tokens=False)["input_ids"]) - 1
        hooks = hook_factory(p, pos)
        s = score_answer_pairs(bundle, [p], [ans], batch_size=1, hooks=hooks)[0]
        total += -s / n_tok
        count += 1
    return total / max(count, 1)
