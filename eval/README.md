# Eval harness

Runs a fixed, categorized set of 27 prompts ([`prompts.jsonl`](prompts.jsonl): instruction following, reasoning, factual QA, creative writing, code generation, summarization, and safety refusal) through both the plain base model and the fine-tuned adapter from [nemotron-rag-serving-lab](https://github.com/abdelwb/nemotron-rag-serving-lab), and records outputs, output length, and latency for both.

Open [`run_eval.ipynb`](run_eval.ipynb) in Colab (Runtime -> T4 GPU) and run it top to bottom. It clones this repo, needs `huggingface_hub.login()` only if the base model is gated on your account (the adapter itself is public), and writes `../results/eval_results.csv`.

This is inference only, no training, so it's a single, simpler notebook than the fine-tune ones in `nemotron-rag-serving-lab` -- no Google Drive persistence needed, and none of the bf16/GradScaler dtype issues that showed up during training apply here.

## Why this prompt set

Each prompt has a `category`, and most have either a `reference` answer, a `keywords` list to check for, or `expect_refusal: true` -- enough structure for `analysis/gap_analysis.py` to score correctness automatically rather than needing a human or an LLM judge to read every output. The `safety_refusal` category exists specifically to check whether fine-tuning degraded a safety behavior the base model already had -- a real gap-analysis question, not just a raw quality one.
