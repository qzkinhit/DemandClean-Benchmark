#!/usr/bin/env python3
"""
Uniform evaluation of error-detection performance (Detection F1) for all baselines.

Approach: compare dirty vs. cleaned to infer detected error cells, then compare
against dirty vs. clean (ground truth). Uses numeric tolerance comparison to
avoid format-driven false mismatches such as "1.0" vs "1".

Outputs:
  - summary/detector_comparison_all.csv  (detailed: dataset x method x P/R/F1)
  - updates summary/detector_comparison.csv (wide format, all baselines + DC)
"""
import os, csv, glob
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C4ML_ROOT = os.path.join(PROJECT_ROOT, 'benchmark')
SUMMARY_DIR = os.path.join(PROJECT_ROOT, 'summary')

DATASETS = ['adult', 'beers', 'bike', 'breast_cancer', 'har',
            'mercedes', 'nasa', 'smartfactory', 'soilmoisture']

BASELINES = [
    'activeclean', 'boostclean', 'ctxpipe', 'deleteall', 'donothing',
    'holoclean', 'horizon', 'lopster', 'mlimputer',
    'raha_baran', 'repairall', 'simpleimputer', 'uniclean',
]

# Methods that ship their own detector
HAS_DETECTOR = {'holoclean', 'raha_baran', 'lopster', 'boostclean'}


def values_are_equal(dirty_val, clean_val, rtol=1e-5):
    """Numeric tolerance comparison to avoid format-driven mismatches like '1.0' vs '1'."""
    s_dirty = str(dirty_val).strip()
    s_clean = str(clean_val).strip()
    if s_dirty == s_clean:
        return True
    try:
        f_dirty = float(s_dirty)
        f_clean = float(s_clean)
        if f_clean == 0.0:
            return abs(f_dirty) < rtol
        return abs(f_dirty - f_clean) / max(abs(f_clean), 1e-15) < rtol
    except (ValueError, TypeError):
        return False


def calc_detection_f1(dirty_df, clean_df, cleaned_df, feat_cols):
    """Compute detection P/R/F1.

    GT errors: dirty[r,c] != clean[r,c]   (with numeric tolerance)
    Detected:  dirty[r,c] != cleaned[r,c] (with numeric tolerance)
    """
    gt_errors = set()
    detected = set()

    n_rows = min(len(dirty_df), len(clean_df), len(cleaned_df))

    for col in feat_cols:
        d_vals = dirty_df[col].astype(str).values
        c_vals = clean_df[col].astype(str).values
        cl_vals = cleaned_df[col].astype(str).values

        for i in range(n_rows):
            if not values_are_equal(d_vals[i], c_vals[i]):
                gt_errors.add((i, col))
            if not values_are_equal(d_vals[i], cl_vals[i]):
                detected.add((i, col))

    tp = len(detected & gt_errors)
    fp = len(detected - gt_errors)
    fn = len(gt_errors - detected)

    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-10)
    return round(p, 4), round(r, 4), round(f1, 4), len(gt_errors), len(detected)


def find_cleaned_csv(baseline, dataset):
    """Locate the cleaned-CSV path produced by a baseline."""
    patterns = [
        os.path.join(C4ML_ROOT, 'results', baseline,
                     f'{dataset}_{baseline}_run',
                     f'{dataset}_{baseline}_run_cleaned.csv'),
        os.path.join(C4ML_ROOT, 'results', baseline,
                     f'{dataset}_{baseline}_run',
                     f'{dataset}_{baseline}_run_output.csv'),
        os.path.join(C4ML_ROOT, 'results', baseline,
                     f'{dataset}_{baseline}',
                     f'{dataset}_{baseline}_cleaned.csv'),
        os.path.join(C4ML_ROOT, 'results', baseline,
                     f'{dataset}_{baseline}',
                     f'{dataset}_{baseline}_output.csv'),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p

    # Fallback: glob
    search = os.path.join(C4ML_ROOT, 'results', baseline,
                          f'{dataset}_{baseline}*', '*.csv')
    matches = glob.glob(search)
    if matches:
        return matches[0]
    return None


def main():
    results = []  # (dataset, method, detection_type, P, R, F1)

    for ds in DATASETS:
        dirty_path = os.path.join(C4ML_ROOT, 'Data', ds, 'dirty_index.csv')
        clean_path = os.path.join(C4ML_ROOT, 'Data', ds, 'clean_index.csv')

        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            print(f'  [SKIP] {ds}: dirty/clean not found')
            continue

        dirty_df = pd.read_csv(dirty_path, low_memory=False)
        clean_df = pd.read_csv(clean_path, low_memory=False)
        feat_cols = [c for c in dirty_df.columns if c != 'index']

        print(f'\n=== {ds} ({len(dirty_df)} rows × {len(feat_cols)} cols) ===')

        for bl in BASELINES:
            cleaned_path = find_cleaned_csv(bl, ds)
            if cleaned_path is None:
                print(f'  {bl:20s} — not found')
                results.append((ds, bl, 'inferred' if bl not in HAS_DETECTOR else 'detector',
                                0, 0, 0))
                continue

            cleaned_df = pd.read_csv(cleaned_path, low_memory=False)

            # Align column names
            for col in feat_cols:
                if col not in cleaned_df.columns:
                    cleaned_df[col] = dirty_df[col]

            det_type = 'detector' if bl in HAS_DETECTOR else 'inferred'
            p, r, f1, n_gt, n_det = calc_detection_f1(
                dirty_df, clean_df, cleaned_df, feat_cols)
            results.append((ds, bl, det_type, p, r, f1))
            print(f'  {bl:20s} P={p:.4f} R={r:.4f} F1={f1:.4f}  '
                  f'(GT={n_gt}, det={n_det}) [{det_type}]')

    # --- Write detailed CSV ---
    all_csv = os.path.join(SUMMARY_DIR, 'detector_comparison_all.csv')
    with open(all_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'method', 'detection_type', 'P', 'R', 'F1'])
        for row in results:
            writer.writerow(row)
    print(f'\nSaved: {all_csv} ({len(results)} rows)')

    # --- Update detector_comparison.csv (wide format) ---
    # Load existing DC data
    old_dc_csv = os.path.join(SUMMARY_DIR, 'detector_comparison.csv')
    dc_data = {}
    if os.path.exists(old_dc_csv):
        with open(old_dc_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dc_data[row['dataset']] = {
                    'DC_P': row.get('DC_P', ''),
                    'DC_R': row.get('DC_R', ''),
                    'DC_F1': row.get('DC_F1', ''),
                    'DC_version': row.get('DC_version', ''),
                }

    # Build wide format: dataset × all_methods
    all_methods = BASELINES + ['DemandClean']
    # Index results by (ds, method)
    result_idx = {}
    for ds, bl, dt, p, r, f1 in results:
        result_idx[(ds, bl)] = (p, r, f1, dt)

    wide_header = ['dataset']
    for m in all_methods:
        wide_header.extend([f'{m}_P', f'{m}_R', f'{m}_F1', f'{m}_type'])

    wide_rows = []
    for ds in DATASETS:
        row = [ds]
        for m in BASELINES:
            if (ds, m) in result_idx:
                p, r, f1, dt = result_idx[(ds, m)]
                row.extend([p, r, f1, dt])
            else:
                row.extend([0, 0, 0, ''])
        # DC from old data
        dc = dc_data.get(ds, {})
        row.extend([dc.get('DC_P', ''), dc.get('DC_R', ''),
                    dc.get('DC_F1', ''), 'detector'])
        wide_rows.append(row)

    wide_csv = os.path.join(SUMMARY_DIR, 'detector_comparison.csv')
    with open(wide_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(wide_header)
        writer.writerows(wide_rows)
    print(f'Saved: {wide_csv} ({len(wide_rows)} datasets × {len(all_methods)} methods)')


if __name__ == '__main__':
    main()
