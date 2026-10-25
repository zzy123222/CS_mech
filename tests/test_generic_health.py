from src.metrics.generic_health import GENERIC_PAIRS


def test_battery_is_reasonably_sized_and_wellformed():
    assert len(GENERIC_PAIRS) >= 12
    for prompt, answer in GENERIC_PAIRS:
        assert isinstance(prompt, str) and isinstance(answer, str)
        assert prompt.strip() and answer.strip()
        assert answer.startswith(" ")


def test_battery_covers_distinct_skills():
    joined = " ".join(p for p, _ in GENERIC_PAIRS).lower()
    for marker in ["capital", "sequence", "opposite", "equals", "copy", "letter"]:
        assert marker in joined
