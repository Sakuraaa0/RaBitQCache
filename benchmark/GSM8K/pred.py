# Prediction script for GSM8K-CoT benchmark with sparse attention methods.
# Dataset: ankner/gsm8k-CoT (grade-school math with chain-of-thought)

import os
import json
import re
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from rabitqcache.pyimpl import enable_sparse_attention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HF model")
    parser.add_argument("--algo-config-path", type=str, required=True, help="Sparse attention config JSON")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing GSM8K JSONL files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save prediction JSONL files")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--max_gen", type=int, default=4096,
                        help="Max tokens to generate")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max number of samples to evaluate (-1 for all)")
    parser.add_argument("--prompt_mode", type=str, default="cot",
                        choices=["cot", "direct"],
                        help="Prompt mode: cot (chain-of-thought) or direct")
    return parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# 8-shot CoT examples from lm-evaluation-harness (gsm8k-cot.yaml)
FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
]


def build_prompt(question, model_name, prompt_mode="cot"):
    """Build prompt following lm-evaluation-harness gsm8k-cot format (8-shot)."""
    if prompt_mode == "cot":
        # 8-shot CoT prompt: "Q: ...\n\nA: ..." for each example
        parts = []
        for ex in FEWSHOT_EXAMPLES:
            parts.append(f"Q: {ex['question']}\n\nA: {ex['answer']}")
        # Append the actual question
        parts.append(f"Q: {question}\n\nA:")
        prompt = "\n\n".join(parts)
    else:
        prompt = f"Q: {question}\n\nA:"

    return prompt


def normalize_number(s):
    """Normalize a number string: remove commas, trailing zeros, etc."""
    s = s.strip().replace(",", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def extract_answer(text):
    """Extract numerical answer from model output.

    Following lm-evaluation-harness priority:
    1. "The answer is X." (strict, matches 8-shot CoT format)
    2. \\boxed{X}
    3. "#### X" (GSM8K native format)
    4. Flexible: last number-like pattern (fallback)
    """
    # Pattern 1 (strict): "The answer is X." — matches lm-eval-harness gsm8k-cot
    match = re.search(r'[Tt]he\s+(?:final\s+)?answer\s+is\s+(\-?[0-9\.\,]+)', text)
    if match:
        return normalize_number(match.group(1))

    # Pattern 2: \boxed{...}
    boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed_matches:
        return normalize_number(boxed_matches[-1])

    # Pattern 3: "#### X" (GSM8K native format)
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if match:
        return normalize_number(match.group(1))

    # Pattern 4 (flexible fallback): last number-like token
    # Matches lm-eval-harness "flexible-extract" filter
    numbers = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    if numbers:
        # Take last match, pick the non-empty group
        last = numbers[-1]
        val = last[0] if last[0] else last[1]
        return normalize_number(val.replace("$", ""))

    return ""


def load_model_and_tokenizer(model_path, algo_config_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = model.eval()

    with open(algo_config_path, "r") as f:
        algo_config = json.load(f)

    budget_info, score_info = enable_sparse_attention(
        model,
        sparse_config=algo_config,
        enable_budget_info=True,
        enable_score_info=True,
    )

    return model, tokenizer, algo_config, budget_info, score_info


def get_pred(model, tokenizer, data, max_gen, model_name, prompt_mode,
             algo_config, budget_info, score_info):
    """Run inference on GSM8K data using manual prefill + decode loop."""
    preds = []

    selector_cfg = algo_config.get("selector", {}) if isinstance(algo_config, dict) else {}
    selector_type = selector_cfg.get("type") if isinstance(selector_cfg, dict) else None

    for idx, json_obj in enumerate(tqdm(data)):
        if selector_type == "rabitq":
            from rabitqcache.pyimpl.rabitq import reset_rabitq_state
            reset_rabitq_state()
        elif selector_type == "rabitq_no_center":
            from rabitqcache.pyimpl.rabitq_no_center import reset_rabitq_no_center_state
            reset_rabitq_no_center_state()
        torch.cuda.empty_cache()

        question = json_obj["question"]
        gold_answer = normalize_number(str(json_obj["answer"]))
        prompt = build_prompt(question, model_name, prompt_mode)

        # Tokenize the full prompt
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            # Prefill
            output = model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values

            # Decode
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]

            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content.append(pred_token_idx.item())
                if pred_token_idx.item() == tokenizer.eos_token_id:
                    break
                # Stop if model starts generating next question (same as lm-eval-harness until: ['Q:'])
                if len(generated_content) >= 2:
                    tail = tokenizer.decode(generated_content[-2:], skip_special_tokens=True)
                    if "Q:" in tail:
                        break

        pred_text = tokenizer.decode(generated_content, skip_special_tokens=True)
        # Truncate at "Q:" if present (model tried to generate next example)
        if "Q:" in pred_text:
            pred_text = pred_text[:pred_text.index("Q:")]

        # Extract answer
        extracted_answer = extract_answer(pred_text)
        is_correct = extracted_answer == gold_answer

        # Record budget/score info
        avg_budget = budget_info.get_total_avg_budget()
        avg_score = score_info.get_total_avg_score()
        budget_info.reset()
        score_info.reset()

        preds.append({
            "id": idx,
            "question": question,
            "gold_answer": gold_answer,
            "pred_text": pred_text,
            "pred_answer": extracted_answer,
            "correct": is_correct,
            "budget": avg_budget,
            "score_sum": avg_score,
        })

        if (idx + 1) % 50 == 0:
            correct_so_far = sum(1 for p in preds if p["correct"])
            print(f"  [{idx+1}/{len(data)}] Running accuracy: {correct_so_far}/{len(preds)} = {correct_so_far/len(preds)*100:.1f}%")

    return preds


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    # Load model with sparse attention
    model, tokenizer, algo_config, budget_info, score_info = load_model_and_tokenizer(
        args.model_path, args.algo_config_path
    )

    # Load data
    data_file = os.path.join(args.data_dir, f"{args.split}.jsonl")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    data = read_jsonl(data_file)
    print(f"Loaded {len(data)} samples from {data_file}")

    if args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Using first {len(data)} samples")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"GSM8K-CoT Prediction ({args.split} split, {len(data)} samples)")
    print(f"{'='*50}")

    preds = get_pred(
        model, tokenizer, data, args.max_gen, args.model_name,
        args.prompt_mode, algo_config, budget_info, score_info,
    )

    # Save predictions
    pred_file = os.path.join(args.save_dir, f"{args.split}.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(preds)} predictions to {pred_file}")

    # Print summary
    correct = sum(1 for p in preds if p["correct"])
    total = len(preds)
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"{'='*50}")
