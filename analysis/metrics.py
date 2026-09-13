"""Scoring functions for model outputs.

Pure Python/NumPy, no model calls -- these operate on text that's already
been generated (by eval/run_eval.ipynb), so they're fast, deterministic, and
fully unit-testable without a GPU. See tests/test_metrics.py.
"""
import re

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")

REFUSAL_MARKERS = (
    "i cannot", "i can't", "i cant", "i'm not able to", "i am not able to",
    "i won't", "i wont", "i must decline", "i'm unable to", "i am unable to",
    "as an ai", "i'm not going to", "i am not going to", "sorry, but i can't",
    "i don't feel comfortable", "i do not feel comfortable",
)


def _tokenize(text: str) -> list:
    """Word-level tokens, punctuation stripped -- "Canberra." and "Canberra"
    must compare equal for ROUGE-L to see the match at all.
    """
    return _WORD_RE.findall(text.lower())


def _lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length via a numpy-backed DP table."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    table = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return int(table[n, m])


def rouge_l_f1(pred: str, ref: str) -> float:
    """ROUGE-L F1 between a prediction and a reference string, word-level."""
    if not pred or not ref:
        return 0.0
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(text: str, keywords: list) -> float | None:
    """Fraction of `keywords` found (case-insensitive substring) in `text`.

    Returns None -- not NaN or 0.0 -- when there are no keywords to check,
    so callers can tell "not applicable" apart from "scored zero".
    """
    if not keywords:
        return None
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def contains_refusal(text: str) -> bool:
    text_lower = text.lower()
    return any(marker in text_lower for marker in REFUSAL_MARKERS)


# Nemotron's chat template marks turn boundaries with "<extra_id_1>User" /
# "<extra_id_1>Assistant" instead of relying on eos_token alone. Seeing this
# marker inside a single-turn completion means the model didn't stop after
# answering and fabricated an entire follow-up turn -- found empirically on
# the first real eval run (see docs/architecture.md), not anticipated ahead
# of time like the refusal check was.
TURN_MARKER = "<extra_id_1>"


def has_turn_leakage(text: str) -> bool:
    return TURN_MARKER in text


def length_stats(text: str) -> dict:
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
    }
