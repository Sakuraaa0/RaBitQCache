<div align="center">

<h1>
    RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference
    <br><br>
    <b>ICML 2026</b>
    <br><br>
    <a href="https://arxiv.org/abs/2606.31519" target="_blank">
      <img src="https://img.shields.io/badge/Paper%20ArXiv-RaBitQCache-b31b1b.svg" alt="Paper ArXiv: GRACE">
    </a>
  </h1>
</div>




## Overview
Long-context Large Language Model inference is severely bottlenecked by the massive Key-Value (KV) cache, yet existing sparse attention methods often suffer from static fixed-budget (Top-k) retrieval or rely on proxy scores that are computationally expensive and biased. To address these limitations, we propose RaBitQCache, a novel sparse attention framework that utilizes randomized rotated binary quantization and high-throughput binary-INT4 arithmetic to efficiently estimate attention weights. Our proxy score serves as an unbiased estimator with a proven error bound, enabling adaptive Top-p retrieval that dynamically adjusts the token budget based on actual attention sparsity. We further implement a hardware-aware system with asynchronous pipelining and lazy updates to mask overhead. Evaluations demonstrate that RaBitQCache significantly accelerates inference and reduces memory I/O while preserving generation quality compared to state-of-the-art baselines.

## RaBitQCache Accuracy Evaluation

This directory contains the accuracy evaluation code for RaBitQCache. It implements a Python-based framework for evaluating sparse attention methods on long-context and reasoning benchmarks.

Supported benchmarks: **LongBench**, **RULER**, **GSM8K-CoT**(and also AIME25 / passkey).

### Installation

```bash
# Create and activate conda environment
conda create -n rabitqcache python=3.10
conda activate rabitqcache

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

**Note:** Installing `flash-attn` may take several minutes. 

### Running Experiments

All scripts follow the pattern `bash scripts/run_<bench>.sh <algo_config_path> [extra args]` from the `benchmark/` directory.

#### LongBench

```bash
cd benchmark

# Run
CUDA_VISIBLE_DEVICES=0 bash scripts/run_longbench.sh configs/configs_rabitq/rabitq_0.65.json

# Evaluate results
bash scripts/eval_longbench.sh
```

#### RULER

```bash
cd benchmark

# Run all 13 RULER tasks at default seq lengths (65536, 98304)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ruler.sh configs/configs_rabitq/rabitq_0.65.json

# Or specify a single sequence length
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ruler.sh configs/configs_rabitq/rabitq_0.65.json 4096

# Re-evaluate existing predictions without rerunning
bash scripts/eval_ruler.sh results_ruler/Meta-Llama-3.1-8B-Instruct/rabitq_0.65
```

#### GSM8K-CoT

```bash
cd benchmark

# Full test set (8-shot CoT prompting)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gsm8k.sh configs/configs_rabitq/rabitq_0.65.json

# Args: <algo_config_path> [max_samples=-1] [max_gen=4096] [prompt_mode=cot|direct]
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gsm8k.sh configs/config_full.json 200
```


### Configuration

#### RaBitQ Configuration Example

```json
{
  "description": "RabitQ-based sparse attention configuration",
  "selector": {
    "type": "rabitq",
    "quantize_interval": 128,
    "top_p": 0.95
  },
  "skip_first_two_layers": true
}
```

**Note on `rabitq.py` vs `rabitq_old.py`:** [rabitqcache/pyimpl/rabitq_old.py](rabitqcache/pyimpl/rabitq_old.py) is the single-GPU implementation; [rabitqcache/pyimpl/rabitq.py](rabitqcache/pyimpl/rabitq.py) is the multi-GPU / pipeline-parallel version. The core algorithm is identical — they only differ in how rotation matrices, centroids, and quantized codes are moved between devices.


### Supported Models

We have validated this implementation on the following models:

- LLaMA-3.1-8B-Instruct
- Longchat-7B-v1.5-32k
- LLaMA-3.1-70B-Instruct

## Efficiency Experiments

The efficiency experiments integrate RaBitQCache into vLLM for system-level performance evaluation.

Note: the current implementation in this section contains a substantial amount of redundant code unrelated to the core method, along with development-time test code, debug switches, and toggles. We plan to clean up and refine this part in the future to retain only what is essential to the method.

### Installation
```bash
cd RabitQCache-Efficiency/vllm

# Install build dependencies
pip install $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml', 'rb'))['build-system']['requires']))")

# Install CUDA and build requirements
pip install -r requirements-cuda.txt
pip install -r requirements-build.txt
pip install -r requirements-common.txt

# Set CMake generator (important for Unix systems)
export CMAKE_GENERATOR="Unix Makefiles"

# Install vLLM with RaBitQ support
pip install -e . --no-build-isolation -v
```

**Requirements:**

- CUDA 12.8+
- Python 3.10+
- NVIDIA GPU with compute capability 8.0+ (recommended: H100, A100, H20)

### Basic RaBitQ Test

```bash
cd RabitQCache-Efficiency

# Run basic inference test
python test/test_rabitq.py
```

### Kernel Benchmark

Compare performance of INT4 × Binary GEMV kernels:

```bash
cd RabitQCache-Efficiency

# Run kernel benchmark
python test/test_kernel_benchmark.py
```

This benchmark compares:

- Original dp4a-optimized kernel
- Packed binary kernel (8× memory bandwidth reduction)
- Warp-level parallel kernel with shared memory


### End-to-End Performance Test

Test with real LongBench samples at different context lengths:

```bash
cd RabitQCache-Efficiency

# Run end-to-end performance test
python test/test_longbench_performance.py --samples 10k,20k,30k
```

## Acknowledgments

During the development of this work, we drew inspiration and reused code from the following open-source projects. We sincerely thank their authors and contributors:

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer.git)
- [vLLM](https://github.com/vllm-project/vllm.git)
- [RaBitQ](https://github.com/gaoj0017/RaBitQ.git)
- [Twilight](https://github.com/tsinghua-ideal/Twilight.git)
- [Quest](https://github.com/mit-han-lab/Quest.git)
- [LMCache](https://github.com/LMCache/LMCache.git)

