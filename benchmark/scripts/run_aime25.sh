#!/bin/bash
# Usage: bash scripts/run_aime25.sh <algo_config_path> [task] [max_gen] [prompt_mode]
# Example: bash scripts/run_aime25.sh configs/configs_rabitq/rabitq_0.65.json
# Example: bash scripts/run_aime25.sh configs/configs_rabitq/rabitq_0.65.json AIME2025-I
# Example: bash scripts/run_aime25.sh configs/config_full.json all 4096 cot
#
# This script:
# 1. Downloads AIME25 data if not present
# 2. Runs prediction with sparse attention via AIME25/pred.py
# 3. Evaluates using AIME25/eval.py

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
DATA_DIR="${BENCHMARK_DIR}/AIME25/data"

algo_cfg_path=$1
if [ -z "$algo_cfg_path" ]; then
    echo "Usage: bash scripts/run_aime25.sh <algo_config_path> [task] [max_gen] [prompt_mode]"
    echo "  task: AIME2025-I, AIME2025-II, or all (default: all)"
    echo "  max_gen: max generation tokens (default: 4096)"
    echo "  prompt_mode: cot or direct (default: cot)"
    exit 1
fi

TASK=${2:-all}
MAX_GEN=${3:-32768}
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
if [ ! -f "${DATA_DIR}/AIME2025-I.jsonl" ] || [ ! -f "${DATA_DIR}/AIME2025-II.jsonl" ]; then
    echo "[Data] Downloading AIME25 dataset..."
    python "${BENCHMARK_DIR}/scripts/download_aime25.py" --save_dir "${DATA_DIR}"
else
    echo "[Data] AIME25 data already exists at ${DATA_DIR}"
fi

# ======================== Step 2: Run prediction ========================
PRED_DIR="${BENCHMARK_DIR}/results_aime25/${MODEL_NAME}/${algo_name}/${TIME}/pred"
mkdir -p "${PRED_DIR}"

echo ""
echo "=============================================="
echo "AIME25 Prediction"
echo "  Model: ${MODEL_NAME}"
echo "  Config: ${algo_cfg_path}"
echo "  Task: ${TASK}"
echo "  Max Gen: ${MAX_GEN}"
echo "  Prompt Mode: ${PROMPT_MODE}"
echo "=============================================="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "${BENCHMARK_DIR}/AIME25/pred.py" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --algo-config-path "${algo_cfg_path}" \
    --data_dir "${DATA_DIR}" \
    --save_dir "${PRED_DIR}" \
    --task "${TASK}" \
    --max_gen "${MAX_GEN}" \
    --prompt_mode "${PROMPT_MODE}"

# ======================== Step 3: Evaluate ========================
echo ""
echo "[Eval] Evaluating predictions..."
python "${BENCHMARK_DIR}/AIME25/eval.py" \
    --pred_dir "${PRED_DIR}"

echo ""
echo "=============================================="
echo "All done! Time: ${TIME}"
echo "Results: ${PRED_DIR}"
echo "=============================================="
