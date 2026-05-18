# Standalone evaluation for RULER predictions, without nemo dependency.
# Adapted from RULER/scripts/eval/evaluate.py

import re
import os
import json
import argparse
import yaml
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ======================== Metrics (from RULER) ========================

def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

TASKS = {
    'niah': {'metric_fn': string_match_all},
    'variable_tracking': {'metric_fn': string_match_all},
    'common_words_extraction': {'metric_fn': string_match_all},
    'freq_words_extraction': {'metric_fn': string_match_all},
    'qa': {'metric_fn': string_match_part},
}

# task name -> base task
TASK_TO_BASE = {
    'niah_single_1': 'niah',
    'niah_single_2': 'niah',
    'niah_single_3': 'niah',
    'niah_multikey_1': 'niah',
    'niah_multikey_2': 'niah',
    'niah_multikey_3': 'niah',
    'niah_multivalue': 'niah',
    'niah_multiquery': 'niah',
    'vt': 'variable_tracking',
    'cwe': 'common_words_extraction',
    'fwe': 'freq_words_extraction',
    'qa_1': 'qa',
    'qa_2': 'qa',
}


def read_jsonl(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def postprocess_pred(predict_str):
    predict_str = predict_str.strip()
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()
    return predict_str


def evaluate_task(pred_file, metric_fn):
    lines = read_jsonl(pred_file)
    preds = []
    refs = []
    for line in lines:
        pred = postprocess_pred(line['pred'])
        ref = line['outputs']
        preds.append(pred)
        refs.append(ref)

    nulls = f'{sum([len(x) == 0 for x in preds])}/{len(preds)}'
    score = metric_fn(preds, refs) if len(refs) > 0 else 0.0
    return score, nulls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='path to prediction jsonl files')
    args = parser.parse_args()

    jsonl_files = [f for f in os.listdir(args.data_dir) if f.endswith('.jsonl')]

    results = {}
    for task_name, base_task in TASK_TO_BASE.items():
        pred_file = os.path.join(args.data_dir, f'{task_name}.jsonl')
        if not os.path.exists(pred_file):
            print(f'Prediction file {task_name}.jsonl not found, skipping.')
            continue

        metric_fn = TASKS[base_task]['metric_fn']
        score, nulls = evaluate_task(pred_file, metric_fn)
        results[task_name] = {'score': score, 'nulls': nulls}
        print(f'{task_name}: {score:.2f} (nulls: {nulls})')

    # Write summary.csv
    if results:
        tasks = list(results.keys())
        scores = [results[t]['score'] for t in tasks]
        nulls = [results[t]['nulls'] for t in tasks]
        dfs = [
            ['Tasks'] + tasks,
            ['Score'] + scores,
            ['Nulls'] + nulls,
        ]
        output_file = os.path.join(args.data_dir, 'summary.csv')
        df = pd.DataFrame(dfs)
        df.to_csv(output_file, index=False)
        print(f'\nSaved eval results to {output_file}')
        print(df)


if __name__ == '__main__':
    main()
