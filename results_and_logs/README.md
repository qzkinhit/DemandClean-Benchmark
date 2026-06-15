# Results and Logs

This directory consolidates all experiment outputs for DemandClean and baseline methods.

## Directory Structure

```
results_and_logs/
├── demandclean/              # DemandClean outputs (9 datasets × multiple versions)
├── reference_strategies/     # NoFix / DeleteAll / ReplaceAll / RepairAll evaluations
├── replaceall_baseline/      # Detailed ReplaceAll with value estimation chain
├── logs/
│   └── demandclean/          # Training and execution logs
└── summary/                  # Cross-method comparison tables
```

## DemandClean Results (`demandclean/`)

Organized by dataset and version:

```
demandclean/{dataset}/{version}/
├── data/
│   ├── {version}_cleaned.csv              # Cleaned output data
│   ├── {version}_cleaned_encoded.npz      # Encoded data (numpy)
│   └── {version}_cleaned_original.csv     # Pre-encoding cleaned data
├── model/
│   ├── {version}_agent.pt                 # Trained DQN agent
│   └── {version}_best_agent.pt            # Best checkpoint
├── evaluation/
│   ├── {version}_evaluation.txt           # Main evaluation report
│   ├── {version}_edr_evaluation.txt       # EDR metrics
│   ├── {version}_hybrid_distance_evaluation.txt
│   ├── {version}_clean_vs_cleaned.csv     # Cell-level comparison
│   ├── {version}_dirty_vs_cleaned.csv     # Change tracking
│   ├── {version}_repair_errors.csv        # Repair error analysis
│   ├── {version}_unrepaired.csv           # Unrepaired errors
│   └── shapley/                           # Shapley value analysis
│       ├── shapley_results.json
│       └── shapley_report.md
└── report/
    ├── {version}_report.json              # Machine-readable report
    ├── {version}_summary.txt              # Human-readable summary
    └── {version}_history.json             # Training history
```

### Versions

| Version | Detector | Agent | Inference |
|---------|----------|-------|-----------|
| v3 | Oracle | Plain DQN | Single-phase |
| v5 | Auto | Dueling DQN | Single-phase |
| v5_ngt | Auto (no ground truth) | Dueling DQN | Single-phase |
| v6 | Auto | Dueling DQN | Two-phase |
| v7 | Auto | Plain DQN | Single-phase |
| v8 | Auto | Plain DQN | Two-phase |

Some datasets have additional tuning variants (e.g., `v5_tuned`, `v5_old`, NASA parameter search `v5_t01`–`v5_t13`).

## Reference Strategies (`reference_strategies/`)

Evaluation of four reference cleaning strategies across all datasets:

- **NoFix**: dirty data as-is (lower bound)
- **DeleteAll**: remove rows with detected errors
- **ReplaceAll**: replace errors with value estimation chain output
- **RepairAll**: replace with ground-truth values (upper bound)

Results in JSON format with per-model (RF, LR, SVM, KNN, DT, GB) metrics.

## ReplaceAll Baseline (`replaceall_baseline/`)

Detailed evaluation of the ReplaceAll strategy using an oracle detector and value estimation chain (VEC). Aligned with DemandClean-Benchmark's evaluation pipeline for fair comparison.

## Summary Tables (`summary/`)

| File | Rows | Description |
|------|------|-------------|
| `demandclean_all_versions_results.csv` | 943 | All DemandClean version metrics across datasets |
| `baseline_evaluation_results_merged.csv` | 2,997 | All 14 baseline methods merged evaluation |
| `detector_comparison.csv` | 10 | Detection P/R/F1 per dataset (wide format) |
| `detector_comparison_all.csv` | 118 | Detailed detection comparison across methods |
| `tolerance_analysis_results.csv` | 86 | Prior and posterior tolerance metrics |
| `replaceall_evaluation_results.csv` | 47 | ReplaceAll per-dataset metrics |
| `shapley_analysis_results.csv` | 46 | Feature contribution attribution |

### Key Columns in Summary CSVs

**demandclean_all_versions_results.csv**:
- `Baseline`: version identifier (e.g., `v5_auto_dueling_single`)
- `Dataset`: dataset name
- `P_do_nothing`, `P_demand_clean`, `P_repair_all`: performance reference points
- `ML_1`–`ML_6`: downstream model scores (RF, LR, SVM, KNN, DT, GB)

**baseline_evaluation_results_merged.csv**:
- `method`: cleaning method name
- `dataset`: dataset name
- `rf_accuracy`, `rf_f1`, etc.: per-model downstream metrics
- `tolerance_prior`, `tolerance_post`: model tolerance metrics
