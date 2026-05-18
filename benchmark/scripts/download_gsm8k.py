"""Download GSM8K-CoT dataset from HuggingFace and convert to JSONL format.

Usage:
    python scripts/download_gsm8k.py [--save_dir GSM8K/data]

Creates:
    <save_dir>/test.jsonl
    <save_dir>/train.jsonl
"""

import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="GSM8K/data",
                        help="Directory to save JSONL files")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from datasets import load_dataset

    for split in ["test", "train"]:
        print(f"Downloading ankner/gsm8k-CoT ({split} split)...")
        ds = load_dataset("ankner/gsm8k-CoT", split=split)

        out_path = os.path.join(args.save_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, row in enumerate(ds):
                obj = {
                    "id": idx,
                    "question": row["question"],
                    "answer": str(row["answer"]).strip(),
                    "response": row.get("response", ""),
                }
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")

        print(f"  Saved {len(ds)} samples to {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
