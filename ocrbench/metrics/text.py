"""Page-level text accuracy metrics."""
from __future__ import annotations

from collections import Counter

from rapidfuzz.distance import Levenshtein

from . import normalize
from .loops import has_loop


def _rate(ref, hyp) -> tuple[int, int]:
    return Levenshtein.distance(ref, hyp), len(ref)


def score_page(gt: str, pred: str, finish_reason: str | None = None) -> dict:
    """Score one page. Returned rates are fractions (0 = perfect); edit counts are kept for micro-averages."""
    dip_ref, dip_hyp = normalize.diplomatic(gt), normalize.diplomatic(pred)
    read_ref, read_hyp = normalize.reading(gt), normalize.reading(pred)
    cer_d_edits, n_chars_d = _rate(dip_ref, dip_hyp)
    cer_r_edits, n_chars = _rate(read_ref, read_hyp)
    ref_words, hyp_words = read_ref.split(), read_hyp.split()
    wer_edits, n_words = _rate(ref_words, hyp_words)

    bow_ref, bow_hyp = Counter(normalize.words(read_ref)), Counter(normalize.words(read_hyp))
    matched = sum((bow_ref & bow_hyp).values())
    n_ref, n_hyp = sum(bow_ref.values()), sum(bow_hyp.values())

    return {
        "cer_dip": cer_d_edits / max(n_chars_d, 1),
        "cer": cer_r_edits / max(n_chars, 1),
        "wer": wer_edits / max(n_words, 1),
        "cer_edits": cer_r_edits,
        "ref_chars": n_chars,
        "bow_recall": matched / n_ref if n_ref else 1.0,
        "bow_precision": matched / n_hyp if n_hyp else (1.0 if not n_ref else 0.0),
        "over_extraction": (n_hyp - matched) / n_hyp if n_hyp else 0.0,
        "len_ratio": len(read_hyp) / max(len(read_ref), 1),
        "empty": not read_hyp,
        "loop": has_loop(pred) or finish_reason == "length",
    }
