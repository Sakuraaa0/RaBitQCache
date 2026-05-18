# Prediction script for AIME 2025 benchmark with sparse attention methods.
# Adapted from RULER/pred.py to work with AIME25 math problems.

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
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing AIME25 JSONL files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save prediction JSONL files")
    parser.add_argument("--task", type=str, default="all", choices=["AIME2025-I", "AIME2025-II", "all"],
                        help="Which AIME2025 exam to evaluate (default: all)")
    parser.add_argument("--max_gen", type=int, default=32768,
                        help="Max tokens to generate (need long output for chain-of-thought)")
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


def build_prompt(problem, model_name, prompt_mode="cot"):
    """Build chat prompt for math problem."""
    if prompt_mode == "cot":
        user_msg = (
            f"Please solve the following math problem step by step. "
            f"The answer is a non-negative integer. "
            f"After your reasoning, provide the final answer as an integer "
            f"in the format \\boxed{{answer}}.\n\n"
            f"Problem: {problem}"
        )
    else:
        user_msg = (
            f"Solve the following math problem. The answer is a non-negative integer. "
            f"Provide only the final integer answer in the format \\boxed{{answer}}.\n\n"
            f"Problem: {problem}"
        )

    if "Llama-3" in model_name:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> "
            f"{user_msg} <|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>"
        )
    elif "Mistral" in model_name:
        prompt = f"[INST] {user_msg} [/INST]"
    elif "Qwen" in model_name:
        prompt = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        # Generic fallback
        prompt = f"User: {user_msg}\nAssistant:"

    return prompt


def extract_answer(text):
    """Extract integer answer from model output.

    Tries multiple patterns:
    1. \\boxed{X}
    2. "the answer is X"
    3. "final answer: X" / "final answer is X"
    4. Last standalone integer in text
    """
    # Pattern 1: \boxed{...}
    boxed_matches = re.findall(r'\\boxed\{(\d+)\}', text)
    if boxed_matches:
        return boxed_matches[-1]

    # Pattern 2: "the answer is X"
    match = re.search(r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(\d+)', text)
    if match:
        return match.group(1)

    # Pattern 3: "final answer: X" or "Final Answer: X"
    match = re.search(r'[Ff]inal\s+[Aa]nswer\s*[:\s]+(\d+)', text)
    if match:
        return match.group(1)

    # Pattern 4: last standalone integer in the text
    integers = re.findall(r'\b(\d+)\b', text)
    if integers:
        return integers[-1]

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
    """Run inference on AIME25 data using manual prefill + decode loop."""
    preds = []

    selector_cfg = algo_config.get("selector", {}) if isinstance(algo_config, dict) else {}
    selector_type = selector_cfg.get("type") if isinstance(selector_cfg, dict) else None

    for json_obj in tqdm(data):
        if selector_type == "rabitq":
            from rabitqcache.pyimpl.rabitq import reset_rabitq_state
            reset_rabitq_state()
        elif selector_type == "rabitq_no_center":
            from rabitqcache.pyimpl.rabitq_no_center import reset_rabitq_no_center_state
            reset_rabitq_no_center_state()
        torch.cuda.empty_cache()

        problem = json_obj["problem"]
        prompt = build_prompt(problem, model_name, prompt_mode)

        # Tokenize the full prompt
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            # Prefill: process the entire prompt
            output = model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values

            # Decode: generate tokens autoregressively
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

        pred_text = tokenizer.decode(generated_content, skip_special_tokens=True)

        # Extract integer answer from model output
        extracted_answer = extract_answer(pred_text)

        # Record budget/score info
        avg_budget = budget_info.get_total_avg_budget()
        avg_score = score_info.get_total_avg_score()
        budget_info.reset()
        score_info.reset()

        preds.append({
            "id": json_obj.get("id", ""),
            "problem": problem,
            "gold_answer": str(json_obj["answer"]),
            "pred_text": pred_text,
            "pred_answer": extracted_answer,
            "correct": str(extracted_answer) == str(json_obj["answer"]),
            "budget": avg_budget,
            "score_sum": avg_score,
        })

        # Print progress
        status = "correct" if preds[-1]["correct"] else "wrong"
        print(f"  Problem {json_obj.get('id', '?')}: pred={extracted_answer}, gold={json_obj['answer']} ({status})")

    return preds


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    # Load model with sparse attention
    model, tokenizer, algo_config, budget_info, score_info = load_model_and_tokenizer(
        args.model_path, args.algo_config_path
    )

    # Determine which tasks to run
    if args.task == "all":
        tasks = ["AIME2025-I", "AIME2025-II"]
    else:
        tasks = [args.task]

    os.makedirs(args.save_dir, exist_ok=True)

    all_preds = []
    for task in tasks:
        task_file = os.path.join(args.data_dir, f"{task}.jsonl")
        if not os.path.exists(task_file):
            print(f"Warning: {task_file} not found, skipping.")
            continue

        data = read_jsonl(task_file)
        print(f"\n{'='*50}")
        print(f"Running {task}: {len(data)} problems")
        print(f"{'='*50}")

        preds = get_pred(
            model, tokenizer, data, args.max_gen, args.model_name,
            args.prompt_mode, algo_config, budget_info, score_info,
        )

        # Save per-task predictions
        pred_file = os.path.join(args.save_dir, f"{task}.jsonl")
        with open(pred_file, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved {len(preds)} predictions to {pred_file}")

        all_preds.extend(preds)

    # Print summary
    if all_preds:
        correct = sum(1 for p in all_preds if p["correct"])
        total = len(all_preds)
        print(f"\n{'='*50}")
        print(f"Overall Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"{'='*50}")
