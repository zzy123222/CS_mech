from src.utils.landmarks import (landmark_token_position, QUESTION_END, FACTS_END,
                                 CUE_END, OPTIONS_END, LAST)


class FakeTok:

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


RAW = ("Use the links: from X reach Y.\n"
       "Graph facts:\n"
       "- N001 -> N002\n"
       "- N003 -> N004\n"
       "A previous quick solver selected option A.\n"
       "Options:\n"
       "A. N005\n"
       "B. N006\n"
       "C. N007\n"
       "D. N008\n"
       "Return only the option letter.")

TEMPLATED = "<|im_start|>user\n" + RAW + "<|im_end|>"


def pos(landmark, raw=RAW, text=TEMPLATED):
    return landmark_token_position(FakeTok(), text, raw, landmark)


def test_positions_are_ordered_and_in_range():
    n_tokens = len(TEMPLATED.split())
    q, f, c, o, l = pos(QUESTION_END), pos(FACTS_END), pos(CUE_END), pos(OPTIONS_END), pos(LAST)
    assert 0 <= q < f < c < o < l == n_tokens - 1


def test_question_end_token_content():
    assert TEMPLATED.split()[pos(QUESTION_END)] == "Y."


def test_missing_landmark_falls_back_to_last():
    raw_no_cue = RAW.replace("A previous quick solver selected option A.\n", "")
    assert pos(CUE_END, raw=raw_no_cue, text=TEMPLATED.replace(
        "A previous quick solver selected option A.\n", "")) == \
        landmark_token_position(FakeTok(), TEMPLATED.replace(
            "A previous quick solver selected option A.\n", ""), raw_no_cue, LAST)


def test_raw_prompt_not_found_falls_back():
    assert landmark_token_position(FakeTok(), "totally unrelated", RAW, FACTS_END) == \
        len("totally unrelated".split()) - 1
