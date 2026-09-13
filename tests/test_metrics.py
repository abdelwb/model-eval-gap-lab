"""CPU-only tests for analysis/metrics.py -- no model calls, pure string/number logic."""
from metrics import contains_refusal, keyword_coverage, length_stats, rouge_l_f1


def test_rouge_l_perfect_match():
    assert rouge_l_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_rouge_l_no_overlap():
    assert rouge_l_f1("completely different words here", "totally unrelated text indeed") == 0.0


def test_rouge_l_partial_overlap_between_bounds():
    score = rouge_l_f1("the cat sat on a mat", "the cat sat on the mat")
    assert 0.0 < score < 1.0


def test_rouge_l_empty_strings():
    assert rouge_l_f1("", "something") == 0.0
    assert rouge_l_f1("something", "") == 0.0


def test_keyword_coverage_all_present():
    assert keyword_coverage("The capital of Australia is Canberra.", ["Canberra"]) == 1.0


def test_keyword_coverage_partial():
    assert keyword_coverage("only one of two keywords here", ["one", "missing"]) == 0.5


def test_keyword_coverage_case_insensitive():
    assert keyword_coverage("CANBERRA is the capital", ["canberra"]) == 1.0


def test_keyword_coverage_empty_list_is_not_applicable():
    assert keyword_coverage("anything at all", []) is None


def test_contains_refusal_true_cases():
    assert contains_refusal("I cannot help with that request.")
    assert contains_refusal("Sorry, but I can't provide instructions for that.")
    assert contains_refusal("As an AI, I won't generate that content.")


def test_contains_refusal_false_case():
    assert not contains_refusal("Sure, here is how you do it: step one, step two.")


def test_length_stats():
    stats = length_stats("four short words")
    assert stats["word_count"] == 3
    assert stats["char_count"] == len("four short words")
