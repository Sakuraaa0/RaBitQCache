# Evaluation script for GSM8K-CoT benchmark.
# Reads prediction JSONL files and computes exact-match accuracy.

import os
import json
import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction JSONL files")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to evaluate")
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


def main():
    args = parse_args()
    pred_file = os.path.join(args.pred_dir, f"{args.split}.jsonl")
    output_csv = args.output_csv or os.path.join(args.pred_dir, "summary.csv")

    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return

    preds = read_jsonl(pred_file)
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    avg_budget = sum(p.get("budget", 0) for p in preds) / total if total > 0 else 0
    avg_score = sum(p.get("score_sum", 0) for p in preds) / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"GSM8K-CoT Evaluation Results ({args.split} split)")
    print(f"{'='*60}")
    print(f"  Accuracy:   {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Avg Budget: {avg_budget:.1f}")
    print(f"  Avg Score:  {avg_score:.4f}")

    # Show some wrong examples
    wrong = [p for p in preds if not p.get("correct", False)]
    if wrong:
        print(f"\n  Sample wrong predictions (showing up to 5):")
        for p in wrong[:5]:
            print(f"    id={p.get('id','?')}: pred={p.get('pred_answer','?')}, gold={p.get('gold_answer','?')}")

    print(f"{'='*60}")

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "correct", "total", "accuracy(%)", "avg_budget", "avg_score"])
        writer.writerow([
            args.split, correct, total,
            f"{correct/total*100:.1f}" if total > 0 else "0.0",
            f"{avg_budget:.1f}",
            f"{avg_score:.4f}",
        ])

    print(f"Summary saved to {output_csv}")


if __name__ == "__main__":
    main()
