#!/usr/bin/env python3
"""
LongBench Performance Test for RabitQ Cache Optimization.

This script tests RabitQ with samples of different TOKEN lengths (10k, 20k, 30k tokens)
selected from the LongBench dataset.

NOTE: LongBench's 'length' field is CHARACTER count, not token count.
We selected samples based on actual TOKEN count using Llama-3.1 tokenizer.

Selected samples (by TOKEN count):
- 10k tokens: triviaqa.jsonl, index 197, ~10000 tokens (single-doc QA)
- 20k tokens: triviaqa.jsonl, index 38, ~20000 tokens (single-doc QA)
- 30k tokens: qmsum.jsonl, index 133, ~30000 tokens (meeting summary QA)
"""

import os
import sys

# =============================================================================
# CRITICAL: All environment variables must be set BEFORE any vLLM imports
# =============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RABITQ_USE_FAISS_INDEX"] = "0"

# Detect debugger
def _is_debugger_active():
    """Check if a debugger is attached."""
    if 'debugpy' in sys.modules:
        return True
    if any('debugpy' in arg for arg in sys.argv):
        return True
    if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
        return True
    return False

_DEBUG_MODE = _is_debugger_active()

if _DEBUG_MODE:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print("[DEBUG MODE] Detected debugger - will use enforce_eager=True")

# Now safe to import vLLM and other modules
import json
import time
import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


# Sample selections from LongBench (selected by TOKEN count, not character count)
# Token counts measured using Llama-3.1 tokenizer
LONGBENCH_SAMPLES = {
    "10k": {
        "file": "longbench/LongBench_data/data/triviaqa.jsonl",
        "index": 197,
        "expected_tokens": 9994,  # Actual token count
        "expected_chars": 6687,   # Character count (LongBench 'length' field)
        "dataset": "triviaqa",
        "description": "Single-doc QA",
    },
    "20k": {
        "file": "longbench/LongBench_data/data/triviaqa.jsonl",
        "index": 38,
        "expected_tokens": 19989,
        "expected_chars": 14340,
        "dataset": "triviaqa",
        "description": "Single-doc QA",
    },
    "30k": {
        "file": "longbench/LongBench_data/data/qmsum.jsonl",
        "index": 133,
        "expected_tokens": 29962,
        "expected_chars": 23788,
        "dataset": "qmsum",
        "description": "Meeting Summary QA",
    },
}


def load_sample(sample_info: dict) -> dict:
    """Load a specific sample from LongBench dataset."""
    filepath = sample_info["file"]
    target_idx = sample_info["index"]

    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    full_path = os.path.join(project_root, filepath)

    with open(full_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == target_idx:
                return json.loads(line)

    raise ValueError(f"Sample not found at index {target_idx} in {filepath}")


def build_prompt(sample: dict) -> str:
    """Build the prompt from LongBench sample."""
    context = sample.get("context", "")
    question = sample.get("input", "")

    # Simple prompt format
    prompt = f"""Please answer the following question based on the given context.

Context:
{context}

Question: {question}

Answer:"""

    return prompt


def run_performance_test(
    model_path: str = "/path/to/your/model",  # e.g., Meta-Llama-3.1-8B-Instruct
    enable_rabitq: bool = True,
    rabitq_b_q: int = 4,
    rabitq_topk: int = 256,
    rabitq_topp: float = 0.95,
    max_model_len: int = 32768,
    sample_keys: list = None,
):
    """
    Run performance test with LongBench samples.

    Args:
        model_path: Path to the model
        enable_rabitq: Whether to enable RabitQ
        rabitq_b_q: Bit width for quantizing rotated queries
        rabitq_topk: Top-K tokens for exact computation
        rabitq_topp: Top-P probability threshold
        max_model_len: Maximum model context length
        sample_keys: List of sample keys to test (e.g., ["10k", "20k", "30k"])
    """
    if sample_keys is None:
        sample_keys = ["10k", "20k", "30k"]

    # Configure LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "10.0"
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")

    # Create KVTransferConfig for LMCache
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    # Disable CUDA Graph
    compilation_config = {"cudagraph_mode": "NONE"}

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
    )

    print("\n" + "=" * 80)
    print(f"LongBench Performance Test - RabitQ: {enable_rabitq}")
    if enable_rabitq:
        print(f"  rabitq_b_q={rabitq_b_q}, rabitq_topk={rabitq_topk}, rabitq_topp={rabitq_topp}")
    print("=" * 80)

    # Initialize LLM
    print("\nInitializing LLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        enable_rabitq=enable_rabitq,
        rabitq_b_q=rabitq_b_q if enable_rabitq else None,
        rabitq_topk=rabitq_topk if enable_rabitq else None,
        rabitq_topp=rabitq_topp if enable_rabitq else None,
        gpu_memory_utilization=0.7,
        max_model_len=max_model_len,
        kv_transfer_config=ktc,
        compilation_config=compilation_config,
        enforce_eager=_DEBUG_MODE,
    )
    print("LLM initialized.")

    results = {}

    for key in sample_keys:
        if key not in LONGBENCH_SAMPLES:
            print(f"Warning: Unknown sample key '{key}', skipping...")
            continue

        sample_info = LONGBENCH_SAMPLES[key]
        print(f"\n{'-' * 60}")
        print(f"Testing {key} sample: {sample_info['description']}")
        print(f"  Dataset: {sample_info['dataset']}")
        print(f"  Expected tokens: {sample_info['expected_tokens']}")

        # Load and build prompt
        sample = load_sample(sample_info)
        prompt = build_prompt(sample)
        char_length = sample.get("length", len(prompt))
        print(f"  Character length: {char_length}")

        # Run inference
        print("  Running inference...")
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed_time = time.time() - start_time

        # Get output
        output_text = outputs[0].outputs[0].text
        output_token_ids = outputs[0].outputs[0].token_ids
        output_tokens = len(output_token_ids)
        expected_answers = sample.get("answers", [])

        print(f"  Time: {elapsed_time:.2f} seconds")
        print(f"  Input tokens: {sample_info['expected_tokens']}")
        print(f"  Output tokens: {output_tokens}")
        print(f"  Output: {output_text[:200]}...")
        print(f"  Expected: {expected_answers[0][:200] if expected_answers else 'N/A'}...")

        results[key] = {
            "sample_info": sample_info,
            "input_tokens": sample_info["expected_tokens"],
            "output_tokens": output_tokens,
            "char_length": char_length,
            "time": elapsed_time,
            "output": output_text,
            "expected": expected_answers,
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Sample':<10} {'Input':<10} {'Output':<10} {'Time (s)':<12} {'Output tok/s':<14}")
    print("-" * 56)
    for key in sample_keys:
        if key in results:
            r = results[key]
            output_toks_per_sec = r["output_tokens"] / r["time"] if r["time"] > 0 else 0
            print(f"{key:<10} {r['input_tokens']:<10} {r['output_tokens']:<10} {r['time']:<12.2f} {output_toks_per_sec:<14.1f}")

    # Clean up
    del outputs
    del llm
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    print("\nLMCache backend cleaned up")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LongBench Performance Test for RabitQ")
    parser.add_argument("--model", type=str,
                        default="/path/to/your/model",
                        help="Path to the model")
    parser.add_argument("--no-rabitq", action="store_true",
                        help="Disable RabitQ (run standard attention)")
    parser.add_argument("--rabitq-b-q", type=int, default=4,
                        help="Bit width for quantizing rotated queries (default: 4)")
    parser.add_argument("--rabitq-topk", type=int, default=256,
                        help="Top-K tokens for exact computation (default: 256)")
    parser.add_argument("--rabitq-topp", type=float, default=0.95,
                        help="Top-P probability threshold (default: 0.95)")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Maximum model context length (default: 32768)")
    parser.add_argument("--samples", type=str, default="10k,20k,30k",
                        help="Comma-separated list of sample keys to test (default: 10k,20k,30k)")

    args = parser.parse_args()

    sample_keys = [s.strip() for s in args.samples.split(",")]

    run_performance_test(
        model_path=args.model,
        enable_rabitq=not args.no_rabitq,
        rabitq_b_q=args.rabitq_b_q,
        rabitq_topk=args.rabitq_topk,
        rabitq_topp=args.rabitq_topp,
        max_model_len=args.max_model_len,
        sample_keys=sample_keys,
    )


if __name__ == "__main__":
    main()
