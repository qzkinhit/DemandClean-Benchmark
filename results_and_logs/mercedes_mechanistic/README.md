# mercedes mechanistic check (why DemandClean beats oracle RepairAll)

Supporting artifact for the question: *is DemandClean's win over oracle full GT-repair on mercedes
a property of the framework or a model-sensitivity artifact?* The check is run on the cleaned
**training split** (only it carries injected errors), authenticity/diversity per Def. 3.4/3.5,
shared `var_dirty` denominator, encoders fit once and shared, 12 zero-variance columns excluded.

## 1. Authenticity (A) and Diversity (V) — `authenticity_diversity.csv`

| Strategy | A_true | R_sam | V (Def. 3.5) |
|----------|--------|-------|--------------|
| NoFix | 0.810 | 1.0 | 1.00 |
| **DemandClean** | 0.836 | 1.0 | 0.76 |
| RepairAll | 1.000 | 1.0 | 0.24 |

The agent issues **no deletions** (R_sam = 1 on both sides), so the entire V gap is variance
retention: full GT-repair contracts mean per-column variance to 24% of the dirty level, while
DemandClean retains 76% at 1.6% GT. The collapse is **distributional, not column-causal** — the
top-|ΔV| columns are near-constant in clean data and carry ~2% cumulative importance
(Spearman(|ΔV|, importance) = 0.147).

## 2. Multi-model robustness — `multi_model_r2.csv`

| Model | R²(DemandClean) | R²(RepairAll) | Δ |
|-------|-----------------|---------------|---|
| RF (selected) | 0.428 | 0.392 | +0.036 |
| GB | 0.446 | 0.434 | +0.012 |
| Ridge | 0.359 | 0.411 | -0.052 |
| KNN | 0.237 | 0.264 | -0.027 |

The win is specific to **tree ensembles** (RF, GB); on linear/instance-based models RepairAll is
better — the signature of a diversity effect, not a universal claim. (LR is excluded: on the fully
repaired one-hot design it degenerates from exact collinearity; the raw value is kept in the CSV
for completeness but not interpreted.)

## 3. Operation counts — `op_counts.csv`

| no-op | GT-repair | delete | VE-replace | total acted | GT% (all cells) |
|-------|-----------|--------|------------|-------------|-----------------|
| 123,298 | 24,851 | 0 | 5,638 | 153,787 | 1.6% |

GT-repair on 24,851 cells = 1.6% of all cells (Table 3), 16% of the cells the agent acts on; the
remaining ~80% are no-ops. No rows are deleted.
