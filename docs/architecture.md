# Architecture & methodology

## Why base vs. fine-tuned, not just the fine-tuned model alone

Evaluating only the fine-tuned adapter answers "is it good." Evaluating it against the base model it started from answers the more useful question: "did fine-tuning actually help, and where did it help or hurt." That's what turns a benchmark number into a gap analysis -- the [nemotron-rag-serving-lab](https://github.com/abdelwb/nemotron-rag-serving-lab) LoRA adapter is a natural subject since it's already trained, public, and this repo's sibling project.

## Prompt set design

27 prompts across 7 categories (`eval/prompts.jsonl`): instruction following, reasoning, factual QA, creative writing, code generation, summarization, and safety refusal. Each prompt carries just enough structure to be scored automatically:

- A `reference` answer, scored with ROUGE-L F1 (word-level longest-common-subsequence overlap).
- A `keywords` list, scored as the fraction present in the output (case-insensitive substring match).
- `expect_refusal: true`, scored as whether the output actually contains a refusal.

Prompts with none of the above (most of `creative_writing`) aren't automatically scorable and are labeled `not_scored` rather than given a fabricated number -- length and latency are still recorded for them, just not a correctness score.

The `safety_refusal` category exists specifically to check for regression, not just capability: did fine-tuning on an unrelated instruction dataset accidentally weaken a refusal behavior the base model already had. `analysis/gap_analysis.py` flags this case (`safety_regression`) separately from an ordinary score drop, because it's a materially more important finding.

## Gap analysis

`analysis/gap_analysis.py` pivots the long-format eval output (one row per variant per prompt) into one row per prompt with both variants' scores side by side, and labels each prompt `improved` / `regressed` / `unchanged` / `not_scored` / `safety_regression` based on the score delta (see `REGRESSION_THRESHOLD` in that file). It writes both a per-prompt detail CSV and a per-category summary CSV; the dashboard consumes both.

## Dashboard

`dashboard/build_dashboard.py` renders one static HTML page with an executive-summary section (plain-language, aimed at a non-technical reader) above a technical section (category tables, a base-vs-finetuned bar chart, a latency comparison, and the actual worst regressions and best improvements with their raw outputs). The safety-regressions section always renders, with an explicit "none found" state, so its absence reads as "checked, clean" rather than "not built."

To publish it as a live page instead of a local file: enable GitHub Pages for this repo (Settings -> Pages -> Deploy from a branch -> `main`, folder `/dashboard/output`) once `output/index.html` has been generated from a real run and committed.

## Running it end to end

1. Open [`eval/run_eval.ipynb`](../eval/run_eval.ipynb) in Colab (T4 GPU). Writes `results/eval_results.csv`.
2. `python analysis/gap_analysis.py` -- writes `results/gap_report.csv` and `results/gap_report_detail.csv`.
3. `python dashboard/build_dashboard.py` -- writes `dashboard/output/index.html`.
4. `pytest tests/` -- the golden-regression tests in `tests/test_golden_regression.py` now have real data to gate on instead of skipping.

## Git workflow

This repo's history uses rebase-first feature branches rather than merge commits (see the branch list on GitHub) -- ordinary work happens on a branch, gets rebased onto the current `main` before it lands, and merges as a fast-forward. To do the same on your own fork:

```bash
git checkout -b feat/my-change
# ... commit your work ...
git fetch origin
git rebase origin/main
git checkout main && git merge --ff-only feat/my-change
```

Commits on `main` are also GPG-signed. Signing is tied to your own identity, so it isn't something to copy from this repo -- generate your own key and register it with GitHub if you don't have one already:

```bash
gpg --full-generate-key                       # if you don't already have a key
gpg --list-secret-keys --keyid-format=long    # find the key ID
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true
gpg --armor --export <KEY_ID>                 # paste this into GitHub -> Settings -> SSH and GPG keys
```
