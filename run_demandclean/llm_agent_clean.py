"""LLM cleaning baseline via Claude subagents (no external API key needed).

防作弊设计:批文件只含 dirty 的 **feature 列**(排除 index 和 label_col),
不含 clean 答案、不含 label。agent 纯推理清洗,prompt 禁止联网/查库。

两个模式:
  prepare: 把数据集的待清洗行分批写成 JSON(供 workflow 的 agent 读取清洗)
  collect: 把 agent 清洗结果汇总成 results/llm_baseline/<dataset>_cleaned_by_llm.csv

中间的清洗由 llm_clean_workflow(Workflow) 完成:每个 agent 读 batch_<i>.json,
清洗后写 cleaned_<i>.json。

用法:
  python run_demandclean/llm_agent_clean.py prepare --dataset flights --batch 40
  (Workflow 清洗各批 → cleaned_<i>.json)
  python run_demandclean/llm_agent_clean.py collect --dataset flights
"""
import argparse
import glob
import json
import os

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MISS = {'', 'empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null', 'None', '__NULL__'}
WORK = '/tmp/llm_clean'

DATASET_DESC = {
    'hospitals': 'US hospital quality-measure records (provider number, hospital name/address, city, state, zipcode, phone, type, owner, measure codes/names, scores).',
    'flights': 'Airline flight schedule records: src, flight code, scheduled/actual departure and arrival clock times like "07:45 PM".',
    'soccer': 'Soccer player records: name, surname, birthyear, birthplace, position, team, city, stadium, season, manager.',
    'rayyan': 'Academic article metadata: title, journal title/abbreviation/issn, volume, issue, created date, pagination, author list.',
    'beers': 'Beer records: id, name, style, ounces, abv, ibu, brewery id/name, city, state.',
    'adult': 'US census person records: age, workclass, education, marital status, occupation, relationship, race, gender, capital gain/loss, hours, country.',
    'tax': 'US tax records: first/last name, gender, area code, phone, city, state, zip, marital status, has-child, salary, single/married/child exemptions.',
    'bike': 'Bike-sharing hourly records with weather and time features (numeric).',
    'breast_cancer': 'Breast-cancer diagnostic measurements (integer cell features 1-10).',
    'mercedes': 'Mercedes manufacturing test records (anonymized categorical X0.. and binary features).',
    'nasa': 'NASA airfoil self-noise measurements (frequency, angle, chord length, velocity, thickness).',
    'smartfactory': 'Smart-factory sensor readings used to predict machine failure (numeric).',
    'soilmoisture': 'Hyperspectral soil-moisture spectral band measurements (numeric).',
}


def get_label_col(dataset):
    import sys
    sys.path.insert(0, _ROOT)
    from run_demandclean.run_demandclean_base import DATASETS
    return DATASETS.get(dataset, {}).get('label_col')


def prepare(dataset, batch_size, error_rows_only):
    ddir = os.path.join(_ROOT, 'data', dataset)
    dirty = pd.read_csv(os.path.join(ddir, 'dirty_index.csv'), dtype=str, keep_default_na=False)
    dirty.columns = [c.strip().strip('﻿') for c in dirty.columns]
    label = get_label_col(dataset)
    feature_cols = [c for c in dirty.columns if c not in ('index', label)]  # 排除 label 防泄露

    if error_rows_only:
        mask = dirty[feature_cols].apply(lambda r: any(str(v).strip() in MISS for v in r), axis=1)
        rows_idx = list(dirty[mask].index)
    else:
        rows_idx = list(dirty.index)

    wdir = os.path.join(WORK, dataset)
    os.makedirs(wdir, exist_ok=True)
    for f in glob.glob(os.path.join(wdir, '*.json')):
        os.remove(f)

    n_batches = 0
    for bstart in range(0, len(rows_idx), batch_size):
        bidx = rows_idx[bstart:bstart + batch_size]
        rows = [{'index': dirty.at[i, 'index'], **{c: dirty.at[i, c] for c in feature_cols}} for i in bidx]
        with open(os.path.join(wdir, f'batch_{n_batches}.json'), 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False)
        n_batches += 1

    meta = {'dataset': dataset, 'n_batches': n_batches, 'feature_cols': feature_cols,
            'label_col': label, 'desc': DATASET_DESC.get(dataset, dataset),
            'n_rows_to_clean': len(rows_idx), 'n_total': len(dirty)}
    with open(os.path.join(wdir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[prepare] {dataset}: {n_batches} batches, {len(rows_idx)}/{len(dirty)} rows, feature_cols={feature_cols}")
    print(f"  workdir: {wdir}")
    print(f"  desc: {meta['desc']}")


def collect(dataset):
    ddir = os.path.join(_ROOT, 'data', dataset)
    dirty = pd.read_csv(os.path.join(ddir, 'dirty_index.csv'), dtype=str, keep_default_na=False)
    dirty.columns = [c.strip().strip('﻿') for c in dirty.columns]
    cleaned = dirty.copy()
    cleaned['index'] = cleaned['index'].astype(str)
    idx_pos = {str(v): i for i, v in enumerate(cleaned['index'])}

    wdir = os.path.join(WORK, dataset)
    meta = json.load(open(os.path.join(wdir, 'meta.json')))
    feature_cols = meta['feature_cols']
    n_applied, n_missing_batches = 0, 0
    for b in range(meta['n_batches']):
        cf = os.path.join(wdir, f'cleaned_{b}.json')
        if not os.path.exists(cf):
            n_missing_batches += 1
            continue
        try:
            obj = json.load(open(cf))
        except json.JSONDecodeError:
            n_missing_batches += 1
            continue
        # 两种格式:
        #  (a) sparse-edit: {"edits":[{"index","column","value"}, ...]}  ← 大批首选,只含改动
        #  (b) 全行:        [{"index", col1, col2, ...}, ...]            ← 旧格式,兼容
        fcol_set = set(feature_cols)
        if isinstance(obj, dict) and 'edits' in obj:
            touched = set()
            for e in obj['edits']:
                iv = str(e.get('index', '')).strip()
                col = str(e.get('column', '')).strip()
                val = e.get('value', '')
                if iv in idx_pos and col in fcol_set and str(val).strip() != '':
                    cleaned.iat[idx_pos[iv], cleaned.columns.get_loc(col)] = str(val)
                    touched.add(iv)
            n_applied += len(touched)
        else:
            arr = obj if isinstance(obj, list) else []
            for fixed in arr:
                iv = str(fixed.get('index', '')).strip()
                if iv in idx_pos:
                    pos = idx_pos[iv]
                    for c in feature_cols:
                        if c in fixed and str(fixed[c]).strip() != '':
                            cleaned.iat[pos, cleaned.columns.get_loc(c)] = str(fixed[c])
                    n_applied += 1

    out_dir = os.path.join(_ROOT, 'results', 'llm_baseline')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{dataset}_cleaned_by_llm.csv')
    cleaned.to_csv(out_path, index=False)
    print(f"[collect] {dataset}: applied {n_applied} rows, missing_batches={n_missing_batches}/{meta['n_batches']}")
    print(f"  saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'collect'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--batch', type=int, default=40)
    # 防作弊关键:默认送【所有行】,不告诉 LLM 哪些行有错。--error-rows-only 仅用于调试(会泄露错误位置)。
    ap.add_argument('--error-rows-only', action='store_true',
                    help='[调试用,会泄露错误位置] 只送检测到的错误行')
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare(args.dataset, args.batch, error_rows_only=args.error_rows_only)
    else:
        collect(args.dataset)


if __name__ == '__main__':
    main()
