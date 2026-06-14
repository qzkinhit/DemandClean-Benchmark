# Dataset: flights

Real-world dirty dataset (native errors, NOT synthetically injected by us).

| Item | Value |
|------|-------|
| Source | UniClean/DDPAgent (flight delay) |
| Task | Classification |
| Downstream model | Random Forest |
| Target | `arrival_delay_bucket` |
| Records | 2376 |
| Features | 6 |
| Classes | 3 |

## Files
- `clean_index.csv` — clean reference version (ground truth), with `index` column
- `dirty_index.csv` — dirty version (native errors), with `index` column
- `rules.txt` — DOMAIN + FD rules for detection

## Notes
- Target column holds the **clean ground-truth label** in both clean and dirty files
  (target is a protected column; cleaning does not modify it).
- Missing values are the literal token `empty`.
- Label derived from sched/act arrival times (kept as features). NoFix 0.771 -> FullFix 0.927 (+22% macro-F1).
