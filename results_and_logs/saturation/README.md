# Saturation check: classical real-world cleaning datasets (R3-O5 support)

Why classical cleaning corpora (hospital, rayyan, tax) are poor testbeds for **task-driven
on-demand cleaning**: on most targets the downstream task is already saturated, i.e. the
NoFix → RepairAll gap is near zero, so no cleaning policy can be distinguished. Reproduce with
`python benchmark/tools/saturation_check.py` (unified protocol, same as the main results).

Full per-target numbers are in `saturation_results.csv`. Summary (RF; downstream accuracy for
classification, R² for regression):

## tax — saturated
- `rate` (regression): NoFix R²=0.989, RepairAll R²=0.988, **gap ≈ 0** — even though 93% of rows
  carry a corrupted `rate`, the value is recoverable from `salary`/exemptions, so cleaning the
  target barely moves downstream R². The budget question has nothing to decide here.

## hospital — saturated on meaningful targets
- Across the 13 non-identifier attributes used in turn as the classification target, the
  NoFix → RepairAll accuracy gap stays within **±4.5%** (most are 0); e.g. State/HospitalType/
  Address2 = 0, Condition −0.5%, MeasureCode −2.5%, Sample +4.5%.
- The four high-cardinality identifier columns (ProviderNumber, ZipCode, PhoneNumber, plus
  near-unique HospitalName) are not meaningful classification targets — they degenerate
  (NoFix encodes to a single bucket → 0) and are excluded from the "saturated" reading.

## rayyan — `article_language` carries no error in this column
- In the rayyan release, `article_language` is **identical between dirty and clean** (0 cell
  errors in that column), so repairing it changes nothing. Under the unified protocol the RF
  NoFix→RepairAll gap on this target is 0.36→0.465, and that residual gap comes entirely from
  errors in *other* feature columns (title/journal fields), not from the target.
- Net: classical corpora either saturate the task (tax, most of hospital) or have the target
  itself error-free (rayyan article_language), which is exactly why a high-dimensional,
  downstream-coupled benchmark (REIN) is the appropriate primary testbed.

> Note: the headline numbers here are recomputed under the public `reeval_with_split` protocol.
> Treat `saturation_results.csv` as the source of truth; cite per-target values from it rather
> than from any earlier ad-hoc estimate.
