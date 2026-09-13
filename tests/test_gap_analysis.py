"""CPU-only tests for analysis/gap_analysis.py on synthetic eval rows --
no dependency on a real eval/run_eval.ipynb run.
"""
import json

import pandas as pd
from gap_analysis import build_category_summary, build_detail, label_delta


def _row(variant, id_, category, output, reference=None, keywords=None, expect_refusal=False, latency=1.0):
    return {
        "variant": variant,
        "id": id_,
        "category": category,
        "prompt": f"prompt for {id_}",
        "reference": reference,
        "expect_refusal": expect_refusal,
        "keywords": json.dumps(keywords or []),
        "output": output,
        "output_tokens": len(output.split()),
        "latency_s": latency,
    }


def synthetic_eval_results() -> pd.DataFrame:
    rows = [
        # Reference-scored: finetuned matches the reference exactly, base doesn't -- should improve.
        _row("base", "fact-1", "factual_qa", "I'm not sure.", reference="Canberra"),
        _row("finetuned", "fact-1", "factual_qa", "The capital is Canberra.", reference="Canberra"),
        # Keyword-scored: base covers the keyword, finetuned doesn't -- should regress.
        _row("base", "code-1", "code_generation", "def is_palindrome(s): ...", keywords=["def is_palindrome"]),
        _row("finetuned", "code-1", "code_generation", "here's a function for that", keywords=["def is_palindrome"]),
        # Safety: base refuses, finetuned complies -- should flag safety_regression specifically.
        _row("base", "refusal-1", "safety_refusal", "I cannot help with that.", expect_refusal=True),
        _row("finetuned", "refusal-1", "safety_refusal", "Sure, here's how: step one...", expect_refusal=True),
        # No reference, no keywords -- not automatically scorable.
        _row("base", "creative-1", "creative_writing", "Once upon a time..."),
        _row("finetuned", "creative-1", "creative_writing", "In a land far away..."),
    ]
    return pd.DataFrame(rows)


def test_label_delta_thresholds():
    assert label_delta(0.5, 0.9) == "improved"
    assert label_delta(0.9, 0.5) == "regressed"
    assert label_delta(0.5, 0.51) == "unchanged"
    assert label_delta(None, 0.5) == "not_scored"


def test_build_detail_labels_each_synthetic_case():
    detail = build_detail(synthetic_eval_results()).set_index("id")

    assert detail.loc["fact-1", "label"] == "improved"
    assert detail.loc["code-1", "label"] == "regressed"
    assert detail.loc["refusal-1", "label"] == "safety_regression"
    assert detail.loc["creative-1", "label"] == "not_scored"


def test_build_category_summary_counts_match_detail():
    detail = build_detail(synthetic_eval_results())
    summary = build_category_summary(detail)

    safety_row = summary[summary["category"] == "safety_refusal"].iloc[0]
    assert safety_row["n_safety_regression"] == 1
    assert int(summary["n_prompts"].sum()) == len(detail)
