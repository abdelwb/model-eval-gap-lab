# Dashboard

Turns the gap analysis into a single static HTML page, deliberately aimed at two different readers on one page: an executive-summary section in plain language up top, and category tables/charts plus a per-prompt drill-down below it for a technical reader.

```bash
pip install -r requirements.txt
python build_dashboard.py
```

Writes `output/index.html`. Open it directly in a browser, or see the repo README for hosting it on GitHub Pages once there's real data in it.

The safety-regressions section is deliberately never empty-looking: it always renders, with an explicit "none found" state, so a reader can tell the difference between "checked, and nothing showed up" and "this section wasn't built."
