"""Build a static HTML dashboard from the gap analysis output.

    python build_dashboard.py

Reads results/gap_report.csv and results/gap_report_detail.csv (written by
analysis/gap_analysis.py) and writes dashboard/output/index.html -- a
single self-contained file with an executive summary aimed at a
non-technical reader, and a technical drill-down below it. Open it directly
in a browser, or serve dashboard/output/ as a static site (see the repo
README for a GitHub Pages option).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TOP_N = 5


def load_reports():
    summary_path = RESULTS_DIR / "gap_report.csv"
    detail_path = RESULTS_DIR / "gap_report_detail.csv"
    if not summary_path.exists() or not detail_path.exists():
        sys.exit(
            "gap_report.csv / gap_report_detail.csv not found.\n"
            "Run analysis/gap_analysis.py first -- see docs/architecture.md."
        )
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    if summary.empty:
        sys.exit("gap_report.csv is empty -- run eval/run_eval.ipynb and analysis/gap_analysis.py first.")
    return summary, detail


def build_summary_text(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    n_prompts = len(detail)
    n_improved = int(summary["n_improved"].sum())
    n_regressed = int(summary["n_regressed"].sum())
    n_unchanged = int(summary["n_unchanged"].sum())
    n_safety = int(summary["n_safety_regression"].sum())
    n_leakage = int(summary["n_turn_completion_regression"].sum())

    overall_base = summary["base_mean_score"].mean()
    overall_finetuned = summary["finetuned_mean_score"].mean()
    direction = "improved" if overall_finetuned > overall_base else "declined"

    lines = [
        f"Across {n_prompts} evaluation prompts, the fine-tuned model's average score "
        f"{direction} from {overall_base:.2f} to {overall_finetuned:.2f} compared to the base model.",
        f"{n_improved} prompts got better, {n_regressed} got worse, and {n_unchanged} stayed about the same.",
    ]
    if n_safety > 0:
        lines.append(
            f"WARNING: {n_safety} prompt(s) where the base model correctly refused an unsafe "
            "request but the fine-tuned model did not -- see the Safety Regressions table below "
            "before using this adapter for anything beyond this demo."
        )
    else:
        lines.append("No safety-refusal regressions were found in this prompt set.")
    if n_leakage > 0:
        lines.append(
            f"WARNING: {n_leakage} prompt(s) where the fine-tuned model answered correctly, then "
            "fabricated an entire follow-up conversation turn the base model never invented -- see "
            "the Turn-Completion Regressions table below."
        )
    else:
        lines.append("No turn-completion regressions (fabricated follow-up turns) were found.")
    return " ".join(lines)


def build_category_chart(summary: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base", x=summary["category"], y=summary["base_mean_score"]))
    fig.add_trace(go.Bar(name="Fine-tuned", x=summary["category"], y=summary["finetuned_mean_score"]))
    fig.update_layout(
        barmode="group",
        title="Mean score by category: base vs. fine-tuned",
        yaxis_title="Mean score (0 to 1)",
        xaxis_title="Category",
        template="plotly_white",
        height=420,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_latency_chart(summary: pd.DataFrame) -> str:
    fig = make_subplots()
    fig.add_trace(go.Bar(name="Base", x=summary["category"], y=summary["base_mean_latency_s"]))
    fig.add_trace(go.Bar(name="Fine-tuned", x=summary["category"], y=summary["finetuned_mean_latency_s"]))
    fig.update_layout(
        barmode="group",
        title="Mean generation latency by category",
        yaxis_title="Seconds",
        xaxis_title="Category",
        template="plotly_white",
        height=380,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def rows_to_html_table(df: pd.DataFrame, columns: list) -> str:
    if df.empty:
        return "<p><em>None found.</em></p>"
    header = "".join(f"<th>{c}</th>" for c in columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_html(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    exec_summary = build_summary_text(summary, detail)
    category_chart = build_category_chart(summary)
    latency_chart = build_latency_chart(summary)

    detail_cols = ["id", "category", "base_score", "finetuned_score", "delta", "label"]
    regressions = detail[detail["label"].isin(["regressed", "safety_regression"])].sort_values("delta")
    improvements = detail[detail["label"] == "improved"].sort_values("delta", ascending=False)
    safety = detail[detail["label"] == "safety_regression"]
    leakage = detail[detail["label"] == "turn_completion_regression"]

    regressions_table = rows_to_html_table(regressions.head(TOP_N), detail_cols)
    improvements_table = rows_to_html_table(improvements.head(TOP_N), detail_cols)
    safety_table = rows_to_html_table(
        safety, ["id", "category", "base_output", "finetuned_output"]
    )
    leakage_table = rows_to_html_table(
        leakage, ["id", "category", "base_output", "finetuned_output"]
    )
    summary_table = rows_to_html_table(summary, list(summary.columns))

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Model Eval Gap Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 0; padding: 2rem;
          background: #f7f7f8; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .subtitle {{ color: #666; margin-top: 0; }}
  section {{ background: white; border-radius: 8px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .exec-summary {{ font-size: 1.05rem; line-height: 1.6; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.7rem; border-bottom: 1px solid #eee; }}
  th {{ background: #fafafa; }}
  .warning {{ background: #fff4e5; border-left: 4px solid #e07b00; padding: 1rem; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>Model Evaluation Gap Report</h1>
  <p class="subtitle">nvidia/Nemotron-Mini-4B-Instruct (base) vs. abdelwb/nemotron-mini-4b-daring-anteater-lora (fine-tuned)</p>

  <section>
    <h2>Executive summary</h2>
    <p class="exec-summary">{exec_summary}</p>
  </section>

  <section>
    <h2>Score by category</h2>
    {category_chart}
  </section>

  <section>
    <h2>Latency by category</h2>
    {latency_chart}
  </section>

  <section>
    <h2>Category summary (technical)</h2>
    {summary_table}
  </section>

  <section>
    <h2>Biggest regressions</h2>
    {regressions_table}
  </section>

  <section>
    <h2>Biggest improvements</h2>
    {improvements_table}
  </section>

  <section class="warning">
    <h2>Safety regressions</h2>
    {safety_table}
  </section>

  <section class="warning">
    <h2>Turn-completion regressions</h2>
    <p>The fine-tuned model answered correctly, then fabricated an entire follow-up conversation
    turn (a "<code>&lt;extra_id_1&gt;User</code>" marker in its own output) that the base model
    never invented for the same prompt. Caught here even on prompts with no reference/keywords to
    score against.</p>
    {leakage_table}
  </section>
</body>
</html>
"""


def main():
    summary, detail = load_reports()
    html = build_html(summary, detail)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
