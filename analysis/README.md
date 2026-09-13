# Gap analysis

Turns the raw base-vs-finetuned outputs in `../results/eval_results.csv` into an actual error/gap analysis, not just a leaderboard number.

```bash
pip install -r requirements.txt
python gap_analysis.py
```

## What it does

- [`metrics.py`](metrics.py) scores each output one of three ways, depending on what the prompt gives it to check against: ROUGE-L F1 against a reference answer, keyword coverage, or (for the `safety_refusal` category) whether the response actually contains a refusal.
- [`gap_analysis.py`](gap_analysis.py) pivots the long-format eval results (one row per variant per prompt) into one row per prompt with both variants' scores side by side, labels each prompt `improved` / `regressed` / `unchanged` / `not_scored`, and flags a `safety_regression` specifically when the base model correctly refused something and the fine-tuned model didn't.
- Writes two files: `results/gap_report_detail.csv` (per-prompt) and `results/gap_report.csv` (per-category aggregate) -- the dashboard reads both.

`REGRESSION_THRESHOLD` in `gap_analysis.py` controls how large a score delta has to be before it counts as a real change instead of scoring noise.
