"""Detect degenerate repetition ("looping"), a common VLM OCR failure mode.

A model that loops keeps emitting the same phrase until it hits max_tokens.
CER punishes this, but we also count it separately because it is a distinct
operational risk at scale: looping pages are slow and expensive as well as wrong.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+", re.U)


def longest_repeat_run(tokens: list[str], max_period: int = 12) -> tuple[int, int]:
    """Return (period, repeats) of the longest back-to-back repetition of any n-gram up to max_period."""
    best = (0, 1)
    n = len(tokens)
    for k in range(1, max_period + 1):
        i = 0
        while i + 2 * k <= n:
            reps = 1
            while i + (reps + 1) * k <= n and tokens[i + reps * k: i + (reps + 1) * k] == tokens[i: i + k]:
                reps += 1
            if reps > 1 and reps * k > best[0] * best[1]:
                best = (k, reps)
            i += k * reps if reps > 1 else 1
    return best


def has_loop(text: str, min_tokens_covered: int = 40, min_repeats: int = 6) -> bool:
    tokens = _TOKEN.findall(text.lower())
    if len(tokens) < min_tokens_covered:
        return False
    period, reps = longest_repeat_run(tokens)
    return reps >= min_repeats and period * reps >= min_tokens_covered
