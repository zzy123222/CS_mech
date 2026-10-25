from __future__ import annotations


QUESTION_END = "question_end"
FACTS_END = "facts_end"
CUE_END = "cue_end"
OPTIONS_END = "options_end"
LAST = "last"

LANDMARKS = [LAST, QUESTION_END, FACTS_END, CUE_END, OPTIONS_END]

_CUE_PREFIXES = ("A previous ", "A quick ", "An unverified ", "Authoritative record: ")


def _line_end_offsets(raw: str) -> list[int]:
    ends, pos = [], 0
    for line in raw.split("\n"):
        pos += len(line)
        ends.append(pos)
        pos += 1
    return ends


def _landmark_line_index(raw: str, landmark: str) -> int | None:
    lines = raw.split("\n")
    if landmark == QUESTION_END:
        return 0
    if landmark == FACTS_END:
        target = -1
        for i, line in enumerate(lines):
            if line.startswith("- "):
                target = i
        return target if target >= 0 else None
    if landmark == CUE_END:
        target = -1
        for i, line in enumerate(lines):
            if line.startswith(_CUE_PREFIXES):
                target = i
        return target if target >= 0 else None
    if landmark == OPTIONS_END:
        for i, line in enumerate(lines):
            if line.strip() == "Options:" and i + 4 < len(lines):
                return i + 4
        return None
    return None


def landmark_token_position(tokenizer, prompt_text: str, raw_prompt: str,
                            landmark: str) -> int:
    n_last = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]) - 1
    if landmark == LAST:
        return max(0, n_last)
    li = _landmark_line_index(raw_prompt, landmark)
    if li is None:
        return max(0, n_last)
    char_end = _line_end_offsets(raw_prompt)[li]
    base = prompt_text.find(raw_prompt)
    if base < 0:
        return max(0, n_last)
    prefix = prompt_text[: base + char_end]
    idx = len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) - 1
    return max(0, min(idx, n_last))
