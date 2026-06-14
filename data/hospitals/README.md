# Dataset: hospitals

Real-world dirty dataset (native errors, NOT synthetically injected by us).

| Item | Value |
|------|-------|
| Source | UniClean/DDPAgent (Hospital governance benchmark) |
| Task | Classification |
| Downstream model | Random Forest |
| Target | `Condition` |
| Records | 1000 |
| Features | 18 |
| Classes | 5 |

## Files
- `clean_index.csv` — clean reference version (ground truth), with `index` column
- `dirty_index.csv` — dirty version (native errors), with `index` column
- `rules.txt` — DOMAIN + FD rules for detection

## Notes
- Target column holds the **clean ground-truth label** in both clean and dirty files
  (target is a protected column; cleaning does not modify it).
- Missing values are the literal token `empty`.
- Cleaning headroom on Condition: NoFix~0.94 -> FullFix~1.0 (+8.3% macro-F1).
