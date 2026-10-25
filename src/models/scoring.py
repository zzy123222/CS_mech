from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from src.hooks.core import temporary_hooks
from src.utils.module_access import get_input_device


def _encode_no_special(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def _prepare_padded_pairs(tok, prompts: list[str], answers: list[str]):
    seqs, answer_starts, answer_lens = [], [], []
    for p, a in zip(prompts, answers):
        pids = _encode_no_special(tok, p)
        aids = _encode_no_special(tok, a)
        if not aids:
            raise ValueError(f"Answer tokenized to zero tokens: {a!r}")
        seqs.append(pids + aids)
        answer_starts.append(len(pids))
        answer_lens.append(len(aids))
    max_len = max(len(x) for x in seqs)
    pad = tok.pad_token_id
    input_ids = torch.full((len(seqs), max_len), pad, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, ids in enumerate(seqs):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    return input_ids, attn, answer_starts, answer_lens


@torch.no_grad()
def score_answer_pairs(bundle, prompts: list[str], answers: list[str], batch_size: int = 8,
                       hooks: list[tuple[torch.nn.Module, object]] | None = None) -> np.ndarray:
    assert len(prompts) == len(answers)
    model, tok = bundle.model, bundle.tokenizer
    device = get_input_device(model)
    scores: list[float] = []
    context = temporary_hooks(hooks) if hooks else nullcontext()
    with context:
        for start in tqdm(range(0, len(prompts), batch_size), desc="sequence scoring", leave=False):
            bp = prompts[start:start + batch_size]
            ba = answers[start:start + batch_size]
            input_ids, attn, starts, lens = _prepare_padded_pairs(tok, bp, ba)
            input_ids, attn = input_ids.to(device), attn.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logp = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
            target = input_ids[:, 1:]
            for i, (s, ln) in enumerate(zip(starts, lens)):
                pred_pos = torch.arange(s - 1, s - 1 + ln, device=device)
                tgt_pos = torch.arange(s, s + ln, device=device)
                vals = logp[i, pred_pos, target[i, tgt_pos - 1]]
                scores.append(float(vals.sum().item()))
    return np.asarray(scores, dtype=np.float64)


def score_choice_matrix(bundle, prompt_texts: list[str], choices: list[list[str]],
                        batch_size: int = 8) -> np.ndarray:
    flat_p, flat_a, offsets = [], [], [0]
    for p, opts in zip(prompt_texts, choices):
        flat_p.extend([p] * len(opts))
        flat_a.extend(opts)
        offsets.append(offsets[-1] + len(opts))
    flat_scores = score_answer_pairs(bundle, flat_p, flat_a, batch_size=batch_size)
    if len(set(len(c) for c in choices)) != 1:
        raise ValueError("score_choice_matrix currently expects same number of choices per row")
    return flat_scores.reshape(len(choices), len(choices[0]))


def single_token_id(tok, text: str) -> int:
    ids = _encode_no_special(tok, text)
    if len(ids) != 1:
        raise ValueError(f"Expected a single token for {text!r}, got ids={ids}")
    return int(ids[0])


def next_token_margin(bundle, prompt_text: str, positive: str, negative: str,
                      hooks=None) -> float:
    model, tok = bundle.model, bundle.tokenizer
    device = get_input_device(model)
    ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    ids = {k: v.to(device) for k, v in ids.items()}
    pos_id, neg_id = single_token_id(tok, positive), single_token_id(tok, negative)
    context = temporary_hooks(hooks) if hooks else nullcontext()
    with torch.no_grad(), context:
        logits = model(**ids, use_cache=False).logits[0, -1].float()
    return float((logits[pos_id] - logits[neg_id]).item())


def mean_next_token_nll(bundle, prompt_continuation_pairs: list[tuple[str, str]], hooks=None) -> float:
    prompts = [p for p, _ in prompt_continuation_pairs]
    answers = [a for _, a in prompt_continuation_pairs]
    scores = score_answer_pairs(bundle, prompts, answers, batch_size=4, hooks=hooks)
    lens = [max(1, len(bundle.tokenizer(a, add_special_tokens=False)["input_ids"])) for a in answers]
    return float(np.mean([-s / l for s, l in zip(scores, lens)]))
