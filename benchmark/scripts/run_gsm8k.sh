#!/bin/bash
# Usage: bash scripts/run_gsm8k.sh <algo_config_path> [max_samples] [max_gen] [prompt_mode]
# Example: bash scripts/run_gsm8k.sh configs/configs_rabitq/rabitq_0.65.json
# Example: bash scripts/run_gsm8k.sh configs/config_full.json 200        # only first 200 samples
# Example: bash scripts/run_gsm8k.sh configs/config_full.json -1 2048 cot
#
# This script:
# 1. Downloads GSM8K-CoT data if not present
# 2. Runs prediction with sparse attention via GSM8K/pred.py
# 3. Evaluates using GSM8K/eval.py

set -e

# ======================== Activate venv ========================
VENV_DIR="/mnt/user-ssd/your_user/rabitq/RabitQCache_accuracy-main/venv"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: ${VENV_DIR}"
fi

# ======================== Config ========================
MODEL_NAME=Meta-Llama-3.1-8B-Instruct
MODEL_PATH=/mnt/user-ssd/your_user/rabitq/models/Llama-3.1-8B-Instruct

BENCHMARK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${BENCHMARK_DIR}/GSM8K/data"

algo_cfg_path=$1
if [ -z "$algo_cfg_path" ]; then
    echo "Usage: bash scripts/run_gsm8k.sh <algo_config_path> [max_samples] [max_gen] [prompt_mode]"
    echo "  max_samples: max number of test samples (-1 for all, default: -1)"
    echo "  max_gen: max generation tokens (default: 2048)"
    echo "  prompt_mode: cot or direct (default: cot)"
    exit 1
fi

MAX_SAMPLES=${2:--1}
MAX_GEN=${3:-4096}
PROMPT_MODE=${4:-cot}

# ======================== Timestamp ========================
month=$(date +"%m")
day=$(date +"%d")
hour=$(date +"%H")
minute=$(date +"%M")
TIME="${month}${day}${hour}${minute}"

# Extract algo config name for output dir
algo_name=$(basename "$algo_cfg_path" .json)

# ======================== Step 1: Download data ========================
if [ ! -f "${DATA_DIR}/test.jsonl" ]; then
    echo "[Data] Downloading GSM8K-CoT dataset..."
    python "${BENCHMARK_DIR}/scripts/download_gsm8k.py" --save_dir "${DATA_DIR}"
else
    echo "[Data] GSM8K-CoT data already exists at ${DATA_DIR}"
fi

# ======================== Step 2: Run prediction ========================
PRED_DIR="${BENCHMARK_DIR}/results_gsm8k/${MODEL_NAME}/${algo_name}/${TIME}/pred"
mkdir -p "${PRED_DIR}"

echo ""
echo "=============================================="
echo "GSM8K-CoT Prediction"
echo "  Model: ${MODEL_NAME}"
echo "  Config: ${algo_cfg_path}"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Max Gen: ${MAX_GEN}"
echo "  Prompt Mode: ${PROMPT_MODE}"
echo "=============================================="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "${BENCHMARK_DIR}/GSM8K/pred.py" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --algo-config-path "${algo_cfg_path}" \
    --data_dir "${DATA_DIR}" \
    --save_dir "${PRED_DIR}" \
    --split test \
    --max_gen "${MAX_GEN}" \
    --max_samples "${MAX_SAMPLES}" \
    --prompt_mode "${PROMPT_MODE}"

# ======================== Step 3: Evaluate ========================
echo ""
echo "[Eval] Evaluating predictions..."
python "${BENCHMARK_DIR}/GSM8K/eval.py" \
    --pred_dir "${PRED_DIR}" \
    --split test

echo ""
echo "=============================================="
echo "All done! Time: ${TIME}"
echo "Results: ${PRED_DIR}"
echo "=============================================="
