# Results Directory

This directory stores the output of each cleaning method.

## Directory Layout

```
results/
├── simpleimputer/          # SimpleImputer output
│   ├── {dataset}_repaired.csv
│   └── {dataset}_summary.txt
├── mlimputer/              # MLImputer output
├── deleteall/              # DeleteAll output
├── donothing/              # DoNothing output
├── baran/                  # Baran_Raha output
├── horizon/                # Horizon output
├── holoclean/              # HoloClean output
├── activeclean/            # ActiveClean output
├── boostclean/             # BoostClean output
├── ctxpipe/                # CtxPipe output
├── uniclean/               # UniClean output
└── lopster/                # Lopster output
```

## File Naming Convention

### Repaired data
```
{dataset}_repaired.csv
```

### Result summary
```
{dataset}_summary.txt
```

### Evaluation result
```
{dataset}_evaluation.json
```

## File Contents

### _repaired.csv
Cleaned/repaired data in the same format as the original dirty data.

### _summary.txt
Includes:
- Execution time
- Execution status
- Cleaning method parameters
- Repair/delete statistics
- Ground-truth usage cost

### _evaluation.json (if present)
Includes unified getScoreML evaluation results:
- Traditional cleaning metrics (accuracy, recall, f1, edr, etc.)
- Downstream task performance (classification/regression/clustering)
- Model tolerance metrics

## Ground-Truth Usage Types

| Type | Description | Representative Methods |
|------|-------------|------------------------|
| Type 1 | Fully automatic, no human involvement | SimpleImputer, MLImputer, HoloClean |
| Type 2 | Requires a small validation set | BoostClean, Baran |
| Type 3 | Iterative interactive labeling | ActiveClean |
