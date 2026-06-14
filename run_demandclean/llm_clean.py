"""LLM-based cleaning baseline.

调用大模型 API 清洗 dirty 数据集,作为一个新的 baseline。
**防作弊**:纯文本 prompt,不给 ground-truth/clean 数据,prompt 明确禁止联网/查库,
只靠模型内部知识 + 行内一致性推理。标准 messages 调用不带工具,不会联网。

可配置(环境变量或参数):
  LLM_CLEAN_MODEL    模型名(默认 claude-sonnet-4-5)
  ANTHROPIC_API_KEY  / ANTHROPIC_BASE_URL  (或用 --api-key/--base-url)
  也可换 OpenAI 兼容端点:--api-style openai --base-url ... --model ...

用法:
  python run_demandclean/llm_clean.py --dataset breast_cancer --limit 50   # 小测试
  python run_demandclean/llm_clean.py --dataset flights                     # 全量
输出: results/llm_baseline/<dataset>_cleaned_by_llm.csv  (按 index 对齐,与 dirty 同列)
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import requests

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MISS = {'', 'empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null', 'None', '__NULL__'}

# 每个数据集的一句话语义(帮助 LLM 理解列,但不泄露答案)
DATASET_DESC = {
    'hospitals': 'US hospital quality-measure records (provider, name, address, measure codes/names, scores).',
    'flights': 'Flight schedule records: source, flight id, scheduled/actual departure and arrival times (e.g. "07:10 AM").',
    'soccer': 'Soccer player records: name, surname, birth year/place, position, team, city, stadium, season, manager.',
    'rayyan': 'Academic article metadata: title, language, journal title/abbreviation/issn, volume, issue, dates, pagination, authors.',
    'beers': 'Beer records: name, style, ounces, abv, ibu, brewery id/name, city, state.',
    'adult': 'US census income records (age, workclass, education, occupation, etc.).',
    'tax': 'Tax records: name, location, marital status, salary, tax rate, exemptions.',
    'bike': 'Bike-sharing hourly counts with weather/time features.',
    'breast_cancer': 'Breast cancer diagnostic measurements (numeric cell features).',
    'mercedes': 'Mercedes manufacturing test records (anonymized categorical/binary features).',
    'nasa': 'NASA airfoil self-noise measurements (frequency, angle, etc.).',
    'smartfactory': 'Smart-factory sensor readings for machine failure.',
    'soilmoisture': 'Hyperspectral soil-moisture spectral band measurements.',
}


def call_anthropic(prompt, model, api_key, base_url, max_tokens=4096, retries=3):
    url = base_url.rstrip('/') + '/v1/messages'
    headers = {'x-api-key': api_key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'}
    body = {'model': model, 'max_tokens': max_tokens, 'messages': [{'role': 'user', 'content': prompt}]}
    for a in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                return r.json()['content'][0]['text']
            if r.status_code in (429, 529, 500, 503):
                time.sleep(2 ** a * 2); continue
            raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
        except requests.RequestException as e:
            if a == retries - 1:
                raise
            time.sleep(2 ** a * 2)
    raise RuntimeError('LLM call failed after retries')


def call_openai(prompt, model, api_key, base_url, max_tokens=4096, retries=3):
    url = base_url.rstrip('/') + '/chat/completions'
    headers = {'Authorization': f'Bearer {api_key}', 'content-type': 'application/json'}
    body = {'model': model, 'max_tokens': max_tokens, 'messages': [{'role': 'user', 'content': prompt}]}
    for a in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            time.sleep(2 ** a * 2)
        except requests.RequestException:
            if a == retries - 1:
                raise
            time.sleep(2 ** a * 2)
    raise RuntimeError('LLM call failed after retries')


def build_prompt(dataset, cols, rows):
    """sparse-edit prompt:整批送入(行含 index),只让模型回报它要改的 cell。
    既防作弊(模型看不到哪些行有错、自主推断),又省 token(只输出改动),还避免大批全行输出截断。"""
    desc = DATASET_DESC.get(dataset, f"records from the '{dataset}' table.")
    rows_json = json.dumps(rows, ensure_ascii=False)
    return f"""You are a strict data-cleaning system. The following are rows from a table of {desc}
Feature columns: {cols}   (each row also carries an "index" field — its row id.)

IMPORTANT: most cells are already CORRECT; only SOME cells contain errors. You are NOT told which rows or \
cells are wrong — judge that yourself from within-row consistency and plausibility. Error kinds: typos, the literal \
token "empty"/"nan"/"NULL" for a missing value, or format inconsistencies. Fix ONLY cells that genuinely look wrong; \
leave correct-looking cells unchanged.

ANTI-CHEATING (critical): use ONLY your own reasoning over these rows. Do NOT search the web, do NOT look up any \
external database or ground-truth, do NOT guess identifiers you cannot infer. For a missing ("empty") value, infer \
the most plausible value from the rest of the SAME row; if truly unknowable, do NOT edit it.

Return ONLY a JSON object of this exact shape (no prose):
{{"edits":[{{"index":"<exact index value from input>","column":"<feature column>","value":"<corrected value>"}}]}}
Include an entry ONLY for cells you actually change. If you change nothing, return {{"edits":[]}}.

Rows:
{rows_json}"""


def parse_edits(text):
    """从模型输出解析 {"edits":[...]} 或裸数组;失败返回 None。"""
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get('edits'), list):
                return obj['edits']
        except json.JSONDecodeError:
            pass
    m = re.search(r'\[.*\]', text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--limit', type=int, default=None, help='只清洗前 N 行(测试用)')
    ap.add_argument('--batch', type=int, default=200,
                    help='每批送入的行数(窄表可 1000;宽表如 mercedes 376 列建议 ~50)')
    ap.add_argument('--model', default=os.environ.get('LLM_CLEAN_MODEL', 'claude-sonnet-4-5'))
    ap.add_argument('--api-style', choices=['anthropic', 'openai'], default='anthropic')
    ap.add_argument('--api-key', default=os.environ.get('ANTHROPIC_API_KEY'))
    ap.add_argument('--base-url', default=os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'))
    # 防作弊:默认送【所有行】,不告诉 LLM 哪些行有错。--error-rows-only 仅调试(会泄露错误位置)。
    ap.add_argument('--error-rows-only', dest='only_error_rows', action='store_true', default=False,
                    help='[调试用,会泄露错误位置] 只送检测到的错误行')
    args = ap.parse_args()

    label_col = None
    try:
        from run_demandclean.run_demandclean_base import DATASETS
        label_col = DATASETS.get(args.dataset, {}).get('label_col')
    except Exception:
        pass

    ddir = os.path.join(_ROOT, 'data', args.dataset)
    dirty = pd.read_csv(os.path.join(ddir, 'dirty_index.csv'), dtype=str, keep_default_na=False)
    dirty.columns = [c.strip().strip('﻿') for c in dirty.columns]
    feature_cols = [c for c in dirty.columns if c not in ('index', label_col)]  # 排除 label 防泄露
    cleaned = dirty.copy()

    # 选要清洗的行
    if args.only_error_rows:
        # 含 MISS 标记的行(缺失);其余真实错误(typo/格式)LLM 也会在这些行里见到
        mask = dirty[feature_cols].apply(lambda r: any(str(v).strip() in MISS for v in r), axis=1)
        target_idx = list(dirty[mask].index)
        # 若缺失行太少,补充一部分行让 LLM 也能改 typo(取前若干)
        if len(target_idx) < len(dirty) * 0.5:
            extra = [i for i in dirty.index if i not in set(target_idx)]
            target_idx = sorted(set(target_idx) | set(extra))  # typo 类错误需全行扫描 → 全量
    else:
        target_idx = list(dirty.index)
    if args.limit:
        target_idx = target_idx[:args.limit]

    has_index = 'index' in dirty.columns
    fcol_set = set(feature_cols)
    caller = call_anthropic if args.api_style == 'anthropic' else call_openai
    n_edits, n_fail_batches = 0, 0
    print(f"[{args.dataset}] model={args.model} rows_to_clean={len(target_idx)} batch={args.batch} (sparse-edit)")
    t0 = time.time()
    for bstart in range(0, len(target_idx), args.batch):
        bidx = target_idx[bstart:bstart + args.batch]
        rows = [{'index': (dirty.at[i, 'index'] if has_index else i),
                 **{c: dirty.at[i, c] for c in feature_cols}} for i in bidx]
        # 行 id → DataFrame 行标签,供 apply 对齐
        id2lbl = {str(dirty.at[i, 'index'] if has_index else i): i for i in bidx}
        prompt = build_prompt(args.dataset, feature_cols, rows)
        try:
            out = caller(prompt, args.model, args.api_key, args.base_url, max_tokens=8192)
            edits = parse_edits(out)
        except Exception as e:
            print(f"  [warn] batch {bstart} call failed: {str(e)[:80]}")
            edits = None
        if edits is None:
            n_fail_batches += 1  # 解析失败 → 该批保持 dirty(不作弊式 fallback)
        else:
            for e in edits:
                iv = str(e.get('index', '')).strip()
                col = str(e.get('column', '')).strip()
                val = e.get('value', '')
                if iv in id2lbl and col in fcol_set and str(val).strip() != '':
                    cleaned.at[id2lbl[iv], col] = str(val)
                    n_edits += 1
        if (bstart // args.batch) % 5 == 0:
            print(f"  progress {bstart + len(bidx)}/{len(target_idx)}  edits={n_edits} fail_batches={n_fail_batches}  {time.time()-t0:.0f}s")

    out_dir = os.path.join(_ROOT, 'results', 'llm_baseline')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.dataset}_cleaned_by_llm.csv')
    cleaned.to_csv(out_path, index=False)
    print(f"[done] total_edits={n_edits} failed_batches={n_fail_batches} elapsed={time.time()-t0:.0f}s  saved: {out_path}")


if __name__ == '__main__':
    main()
