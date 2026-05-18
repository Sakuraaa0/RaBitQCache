#!/usr/bin/env python3
"""Test script for RabitQ cache optimization in vLLM."""

import os
import sys

# =============================================================================
# CRITICAL: All environment variables must be set BEFORE any vLLM imports
# because vLLM spawns subprocesses that inherit the environment at import time
# =============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RABITQ_USE_FAISS_INDEX"] = "0"
# Disable multiprocessing to see profile output directly
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Detect if running under debugger (debugpy, pdb, etc.) and disable torch.compile
# torch.compile/Dynamo cannot trace generators injected by debuggers
def _is_debugger_active():
    """Check if a debugger is attached."""
    # Check for debugpy (VSCode debugger) - check both module and launcher path
    if 'debugpy' in sys.modules:
        return True
    # Check if launched via debugpy launcher
    if any('debugpy' in arg for arg in sys.argv):
        return True
    # Check for pdb
    if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
        return True
    return False

# Global flag for debug mode - used later when creating LLM
_DEBUG_MODE = _is_debugger_active()

if _DEBUG_MODE:
    # Disable multiprocessing to make debugging easier
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print("[DEBUG MODE] Detected debugger - will use enforce_eager=True to disable torch.compile")

# Now safe to import vLLM and other modules

import gc
import torch
from vllm import LLM, SamplingParams
from vllm.config import CacheConfig
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from vllm.config import CUDAGraphMode
import logging


def rabitq_only():
    """Test only RabitQ cache optimization."""
    
    import time
    import json
    
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "10.0"  # 10GB
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")

    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    # Load standard timing if available
    
    prompts = [
                "The future of AI is",
                # "Hello, my name is",
                # "The life is",
                # "Once upon a time,",
        ]
    # prompts = ["Hello, my name is"]

    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=50,  # Generate more tokens to see decode profile
    )
    
    print("\n" + "=" * 80)
    print("Testing RabitQ Cache Optimization")
    print("=" * 80)

    compilation_config = { "cudagraph_mode": "NONE"}

    llm_rabitq = LLM(
        model="/path/to/your/model",  # e.g., Meta-Llama-3.1-8B-Instruct
        tensor_parallel_size=1,
        enable_rabitq=True,
        rabitq_b_q=4,
        rabitq_topk=256,
        rabitq_topp=0.95,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        kv_transfer_config=ktc,
        # disable_log_stats=True,
        compilation_config=compilation_config, 
        enforce_eager=_DEBUG_MODE,  # Disable torch.compile when debugging
    )

    print("Running inference...")
    start_time = time.time()
    responses_rabitq = llm_rabitq.generate(prompts, sampling_params)
    rabitq_time = time.time() - start_time
    print(f"RabitQ attention time: {rabitq_time:.2f} seconds")
    
    # Print results
    for i, response in enumerate(responses_rabitq):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {response.outputs[0].text}")
        print("-" * 40)
     
    # Clean up
    del responses_rabitq
    del llm_rabitq
        # Clean up LMCache backend
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    print("\nLMCache backend cleaned up")
    
    print("\nRabitQ cache test completed!")




if __name__ == "__main__":
    rabitq_only()
