# Dataset: soccer

Real-world dirty dataset (native errors, NOT synthetically injected by us).

| Item | Value |
|------|-------|
| Source | UniClean soccer (10k subset of 200k) |
| Task | Classification |
| Downstream model | Random Forest |
| Target | `manager` |
| Records | 10610 |
| Features | 9 |
| Classes | 15 |

## Files
- `clean_index.csv` — clean reference version (ground truth), with `index` column
- `dirty_index.csv` — dirty version (native errors), with `index` column
- `rules.txt` — DOMAIN + FD rules for detection

## Notes
- Target column holds the **clean ground-truth label** in both clean and dirty files
  (target is a protected column; cleaning does not modify it).
- Missing values are the literal token `empty`.
- NoFix 0.849 -> FullFix 1.0 (+26% macro-F1) on 200k; verify on 10k subset.
