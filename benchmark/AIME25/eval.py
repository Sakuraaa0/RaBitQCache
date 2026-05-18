# Evaluation script for AIME 2025 benchmark.
# Reads prediction JSONL files and computes exact-match accuracy.

import os
import json
import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction JSONL files (AIME2025-I.jsonl, AIME2025-II.jsonl)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save summary CSV (default: <pred_dir>/summary.csv)")
    return parser.parse_args()


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_task(preds):
    """Compute accuracy and average budget for a set of predictions."""
    total = len(preds)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "avg_budget": 0.0, "avg_score": 0.0}

    correct = sum(1 for p in preds if p.get("correct", False))
    avg_budget = sum(p.get("budget", 0) for p in preds) / total
    avg_score = sum(p.get("score_sum", 0) for p in preds) / total

    return {
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total,
        "avg_budget": avg_budget,
        "avg_score": avg_score,
    }


def main():
    args = parse_args()
    pred_dir = args.pred_dir
    output_csv = args.output_csv or os.path.join(pred_dir, "summary.csv")

    tasks = ["AIME2025-I", "AIME2025-II"]
    results = {}
    all_preds = []

    print(f"\n{'='*60}")
    print(f"AIME 2025 Evaluation Results")
    print(f"{'='*60}")

    for task in tasks:
        pred_file = os.path.join(pred_dir, f"{task}.jsonl")
        if not os.path.exists(pred_file):
            print(f"  {task}: not found, skipping")
            continue

        preds = read_jsonl(pred_file)
        result = evaluate_task(preds)
        results[task] = result
        all_preds.extend(preds)

        print(f"\n  {task}:")
        print(f"    Accuracy: {result['correct']}/{result['total']} = {result['accuracy']:.1f}%")
        print(f"    Avg Budget: {result['avg_budget']:.1f}")
        print(f"    Avg Score:  {result['avg_score']:.4f}")

        # Print per-problem details
        for p in preds:
            status = "OK" if p.get("correct", False) else "WRONG"
            print(f"    Problem {p.get('id', '?')}: pred={p.get('pred_answer', '?')}, "
                  f"gold={p.get('gold_answer', '?')} [{status}]")

    # Overall
    if all_preds:
        overall = evaluate_task(all_preds)
        results["Overall"] = overall
        print(f"\n  {'='*40}")
        print(f"  Overall: {overall['correct']}/{overall['total']} = {overall['accuracy']:.1f}%")
        print(f"  Avg Budget: {overall['avg_budget']:.1f}")
        print(f"  Avg Score:  {overall['avg_score']:.4f}")

    print(f"\n{'='*60}")

    # Save summary CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "correct", "total", "accuracy(%)", "avg_budget", "avg_score"])
        for task_name, result in results.items():
            writer.writerow([
                task_name,
                result["correct"],
                result["total"],
                f"{result['accuracy']:.1f}",
                f"{result['avg_budget']:.1f}",
                f"{result['avg_score']:.4f}",
            ])

    print(f"Summary saved to {output_csv}")


if __name__ == "__main__":
    main()
