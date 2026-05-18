# Prediction script for RULER benchmark with sparse attention methods.
# Adapted from LongBench/pred.py to work with RULER's data format.

import os
import json
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
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing task JSONL files (RULER format)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save prediction JSONL files")
    parser.add_argument("--task", type=str, required=True, help="Task name, e.g. niah_single_1")
    parser.add_argument("--max_gen", type=int, default=128, help="Max tokens to generate (overridden by task config if available)")
    parser.add_argument("--subset", type=str, default="validation", help="validation or test")
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


def load_model_and_tokenizer(model_path, algo_config_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
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


def get_pred(model, tokenizer, data, max_gen, algo_config, budget_info, score_info):
    """Run inference on RULER data using manual prefill + decode loop."""
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

        prompt = json_obj["input"]

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

        # Record budget/score info
        avg_budget = budget_info.get_total_avg_budget()
        avg_score = score_info.get_total_avg_score()
        budget_info.reset()
        score_info.reset()

        # RULER evaluation expects these fields:
        # {index, input, outputs, pred, others, truncation, length}
        preds.append({
            "index": json_obj["index"],
            "pred": pred_text,
            "input": json_obj["input"],
            "outputs": json_obj["outputs"],
            "others": json_obj.get("others", {}),
            "truncation": json_obj.get("truncation", -1),
            "length": json_obj.get("length", -1),
            "budget": avg_budget,
            "score_sum": avg_score,
        })

    return preds


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    # Load model with sparse attention
    model, tokenizer, algo_config, budget_info, score_info = load_model_and_tokenizer(
        args.model_path, args.algo_config_path
    )

    # Locate input data file
    task_file = os.path.join(args.data_dir, args.task, f"{args.subset}.jsonl")
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Task data not found: {task_file}")

    data = read_jsonl(task_file)
    print(f"Loaded {len(data)} samples from {task_file}")

    # Determine max generation tokens
    max_gen = args.max_gen

    # Run prediction
    preds = get_pred(model, tokenizer, data, max_gen, algo_config, budget_info, score_info)

    # Save predictions in RULER-compatible format
    os.makedirs(args.save_dir, exist_ok=True)
    pred_file = os.path.join(args.save_dir, f"{args.task}.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(preds)} predictions to {pred_file}")
