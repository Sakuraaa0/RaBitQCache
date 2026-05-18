"""Download AIME 2025 dataset from HuggingFace and convert to JSONL format.

Usage:
    python scripts/download_aime25.py [--save_dir AIME25/data]

Creates:
    <save_dir>/AIME2025-I.jsonl
    <save_dir>/AIME2025-II.jsonl
"""

import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="AIME25/data",
                        help="Directory to save JSONL files")
    parser.add_argument("--source", type=str, default="opencompass/AIME2025",
                        choices=["opencompass/AIME2025", "math-ai/aime25"],
                        help="HuggingFace dataset source")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from datasets import load_dataset

    if args.source == "opencompass/AIME2025":
        for config_name in ["AIME2025-I", "AIME2025-II"]:
            print(f"Downloading {config_name} from {args.source}...")
            ds = load_dataset(args.source, config_name, split="test")

            out_path = os.path.join(args.save_dir, f"{config_name}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for idx, row in enumerate(ds):
                    obj = {
                        "id": idx,
                        "problem": row["question"],
                        "answer": str(row["answer"]).strip(),
                    }
                    json.dump(obj, f, ensure_ascii=False)
                    f.write("\n")

            print(f"  Saved {len(ds)} problems to {out_path}")

    elif args.source == "math-ai/aime25":
        print(f"Downloading from {args.source}...")
        ds = load_dataset(args.source, split="test")

        # math-ai/aime25 has 30 problems total (15 AIME-I + 15 AIME-II)
        # Split by id: 0-14 = AIME-I, 15-29 = AIME-II
        aime_i = []
        aime_ii = []
        for row in ds:
            obj = {
                "id": int(row["id"]) if str(row["id"]).isdigit() else row["id"],
                "problem": row["problem"],
                "answer": str(row["answer"]).strip(),
            }
            if isinstance(obj["id"], int) and obj["id"] < 15:
                aime_i.append(obj)
            else:
                aime_ii.append(obj)

        for name, data in [("AIME2025-I", aime_i), ("AIME2025-II", aime_ii)]:
            out_path = os.path.join(args.save_dir, f"{name}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for obj in data:
                    json.dump(obj, f, ensure_ascii=False)
                    f.write("\n")
            print(f"  Saved {len(data)} problems to {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
