"""Saturation check for classical real-world cleaning datasets (R3-O5 support).

Shows why hospital/rayyan/tax are poor testbeds for task-driven on-demand cleaning:
the downstream task is already saturated, i.e. the NoFix -> RepairAll accuracy/R2 gap is
near zero, so no cleaning policy can be distinguished. Uses the same unified protocol as
the main results (benchmark/tools/reeval_with_split.py): 60/20/20 split (seed 42), encoder
fit on the dirty 60% train, train on dirty/clean train, test on the clean test split.

Data:
  hospital  -> benchmark/Data/hospitals               (in repo)
  rayyan    -> results_and_logs/saturation/data/rayyan (UniClean; copied for reproducibility)
  tax       -> results_and_logs/saturation/data/tax    (10k subset; copied for reproducibility)

Output: results_and_logs/saturation/saturation_results.csv
"""
import os, sys, contextlib, io
import warnings; warnings.filterwarnings('ignore')
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'benchmark', 'tools'))
from reeval_with_split import preprocess_for_ml, demandclean_split, evaluate_ml_split  # noqa: E402

MODEL = 'rf'


def gap(ddir, target, task='classification'):
    dirty = pd.read_csv(os.path.join(ddir, 'dirty_index.csv'))
    clean = pd.read_csv(os.path.join(ddir, 'clean_index.csv'))
    # 对齐行数(子集可能不同)
    common = set(dirty['index'].astype(str)) & set(clean['index'].astype(str))
    dirty = dirty[dirty['index'].astype(str).isin(common)].reset_index(drop=True)
    clean = clean[clean['index'].astype(str).isin(common)].reset_index(drop=True)
    n = len(dirty)
    tr, va, te = demandclean_split(n)
    _, _, e, s, l = preprocess_for_ml(dirty.iloc[tr].reset_index(drop=True), target, 'index', [], 'auto')

    def en(df):
        return preprocess_for_ml(df, target, 'index', [], 'auto',
                                 fitted_encoders=e, fitted_scaler=s, fitted_label_encoder=l)
    Xd, yd = en(dirty); Xc, yc = en(clean)
    mk = 'r2' if task == 'regression' else 'accuracy'

    def q(Xtr, ytr):
        with contextlib.redirect_stdout(io.StringIO()):
            r = evaluate_ml_split(Xtr, ytr, Xc[te], yc[te], task, [MODEL])
        return r.get(MODEL, {}).get(mk)
    nf, ra = q(Xd[tr], yd[tr]), q(Xc[tr], yc[tr])
    return nf, ra


def main():
    rows = []
    # hospital: every attribute in turn as a classification target
    hdir = os.path.join(REPO, 'benchmark', 'Data', 'hospitals')
    hcols = [c for c in pd.read_csv(os.path.join(hdir, 'clean_index.csv'), nrows=1).columns if c != 'index']
    for t in hcols:
        try:
            nf, ra = gap(hdir, t, 'classification')
            if nf is not None and ra is not None:
                rows.append(['hospital', t, 'classification', round(nf, 4), round(ra, 4), round(ra - nf, 4)])
        except Exception as ex:
            rows.append(['hospital', t, 'classification', None, None, f'skip:{str(ex)[:30]}'])
    # rayyan: article_language
    try:
        nf, ra = gap(os.path.join(REPO, 'results_and_logs', 'saturation', 'data', 'rayyan'),
                     'article_language', 'classification')
        rows.append(['rayyan', 'article_language', 'classification', round(nf, 4), round(ra, 4), round(ra - nf, 4)])
    except Exception as ex:
        rows.append(['rayyan', 'article_language', 'classification', None, None, f'err:{str(ex)[:30]}'])
    # tax: rate (regression R2)
    try:
        nf, ra = gap(os.path.join(REPO, 'results_and_logs', 'saturation', 'data', 'tax'),
                     'rate', 'regression')
        rows.append(['tax', 'rate', 'regression', round(nf, 4), round(ra, 4), round(ra - nf, 4)])
    except Exception as ex:
        rows.append(['tax', 'rate', 'regression', None, None, f'err:{str(ex)[:30]}'])

    df = pd.DataFrame(rows, columns=['dataset', 'target', 'task', 'NoFix', 'RepairAll', 'gap'])
    out = os.path.join(REPO, 'results_and_logs', 'saturation', 'saturation_results.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    # 打印 hospital 摘要
    h = df[(df.dataset == 'hospital') & df.gap.apply(lambda x: isinstance(x, float))]
    print(df.to_string(index=False))
    if len(h):
        print(f"\nhospital: max gap over attributes = {h.gap.max():.4f}; "
              f"MeasureCode gap = {h[h.target=='MeasureCode'].gap.values}")
    print(f"\n[saved] {out}")


if __name__ == '__main__':
    main()
