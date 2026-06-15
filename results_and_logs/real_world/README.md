# Real-world datasets: DemandClean vs. baselines vs. LLM cleaning

Held-out evaluation on four real-world datasets with native errors — `beers` (natively dirty),
`flights`, `soccer`, `hospitals` (UniClean real-world benchmark) — none of which calibrated any
part of DemandClean's training injector. This is the generalization check requested in review.

## Protocol

All methods use the **unified split protocol** (`benchmark/tools/reeval_with_split.py`):
60/20/20 split (seed 42), feature encoder fit on the **dirty 60% train split**, train on the
cleaned training split, **test on the clean ground-truth test split**, per-dataset feature set
and downstream model from `benchmark/tools/datasets_config.py`. Two reading notes:

- **DemandClean** numbers are read from each run's `report.json` (the agent's own encoded output,
  same convention as the main paper's Table 3). NoFix / DeleteAll / RepairAll / external cleaners /
  LLM are recomputed by `reeval_with_split.py` on the public data. The two encoders differ but the
  test target is identical (the clean test split), so the columns are directly comparable, exactly
  as in the paper.
- `real_world_results.csv` is an **independent recomputation** under `reeval_with_split.py` and
  agrees with the table below to ~0.004 (the only gap is `beers` NoFix, a rounding-level difference
  in how the literal `empty` token is encoded).

## Headline result (downstream model selected per dataset by validation tolerance)

| Dataset | Model | NoFix | DeleteAll | RepairAll (oracle) | Best external | LLM (API) | **DemandClean** | GT% |
|---------|-------|-------|-----------|--------------------|---------------|-----------|-----------------|-----|
| beers    | RF | .228 | .207 | .295 | .295 (Raha+Baran) | .272 | **.344** | <0.1 |
| flights  | DT | .981 | .908 | 1.00 | .914 (Baran)      | .992 | **.996** | 0.7  |
| soccer   | DT | .902 | .856 | 1.00 | .811 (Horizon)    | .952 | **1.00** | 0    |
| hospitals| RF | 1.00 | .995 | .995 | ~1.0              | .995 | **1.00** | 0    |

- **beers**: DemandClean exceeds even the oracle RepairAll upper bound, and beats the strongest
  external cleaner (Raha+Baran) by +17%.
- **flights / soccer**: DemandClean reaches (soccer) or comes within .004 of (flights) the oracle
  ceiling, while beating the best external cleaner by +.08 / +.19, at <1% / zero ground truth.
- **hospitals**: a saturated task (no cleaning already hits the ceiling); DemandClean recognizes
  there is nothing worth repairing and matches the ceiling at **zero** ground truth.
- **LLM baseline**: a state-of-the-art LLM accessed via API, prompting only (see
  `llm_baseline/`), does not surpass DemandClean on any dataset, while costing far more (per-table
  token cost in `llm_baseline/llm_cost.csv`).

`beers` DeleteAll is taken from the paper's getScoreML protocol (.207); under the strict
oracle-row-deletion variant beers degenerates because nearly every row carries an error in some
column, so `real_world_results.csv` omits that single cell rather than report the degenerate value.

## Files

- `real_world_results.csv` — every method × every downstream model (rf/lr/svm/knn/dt/gb), test metric.
- `llm_baseline/<dataset>_cleaned_by_llm.csv` — the LLM-cleaned tables (aligned by `index`).
- `llm_baseline/llm_cost.csv` — LLM vs. DemandClean accuracy, input tokens, and USD cost per dataset.
- `llm_baseline/prompt_template.txt` — the exact prompt used (anti-leakage / anti-cheating design).
