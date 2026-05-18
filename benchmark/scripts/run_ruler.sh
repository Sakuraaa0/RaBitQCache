#!/bin/bash
# Usage: bash scripts/run_ruler.sh <algo_config_path> [seq_length]
# Example: bash scripts/run_ruler.sh configs/configs_rabitq/rabitq_0.65.json
# Example: bash scripts/run_ruler.sh configs/configs_rabitq/rabitq_0.65.json 4096
#
# This script:
# 1. Uses RULER's prepare.py to generate synthetic data (if not already generated)
# 2. Runs prediction with sparse attention via RULER/pred.py
# 3. Evaluates using RULER's evaluate.py

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

RULER_DIR=/mnt/user-ssd/your_user/rabitq/RULER/scripts
BENCHMARK_DIR="$(cd "$(dirname "$0")/.." && pwd)"

NUM_SAMPLES=25
MODEL_TEMPLATE_TYPE=meta-llama3
TOKENIZER_TYPE=hf

algo_cfg_path=$1
if [ -z "$algo_cfg_path" ]; then
    echo "Usage: bash scripts/run_ruler.sh <algo_config_path> [seq_length]"
    exit 1
fi

# Sequence lengths: use argument or default to all
if [ -n "$2" ]; then
    SEQ_LENGTHS=($2)
else
    SEQ_LENGTHS=(65536 98304)
fi

# All 13 RULER tasks
TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

# tokens_to_generate per base task (from RULER's data/synthetic/constants.py)
declare -A TASK_MAX_GEN
TASK_MAX_GEN[niah]=128
TASK_MAX_GEN[variable_tracking]=30
TASK_MAX_GEN[common_words_extraction]=120
TASK_MAX_GEN[freq_words_extraction]=50
TASK_MAX_GEN[qa]=32

# Map task name -> base task for max_gen lookup
declare -A TASK_TO_BASE
TASK_TO_BASE[niah_single_1]=niah
TASK_TO_BASE[niah_single_2]=niah
TASK_TO_BASE[niah_single_3]=niah
TASK_TO_BASE[niah_multikey_1]=niah
TASK_TO_BASE[niah_multikey_2]=niah
TASK_TO_BASE[niah_multikey_3]=niah
TASK_TO_BASE[niah_multivalue]=niah
TASK_TO_BASE[niah_multiquery]=niah
TASK_TO_BASE[vt]=variable_tracking
TASK_TO_BASE[cwe]=common_words_extraction
TASK_TO_BASE[fwe]=freq_words_extraction
TASK_TO_BASE[qa_1]=qa
TASK_TO_BASE[qa_2]=qa

# ======================== Timestamp ========================
month=$(date +"%m")
day=$(date +"%d")
hour=$(date +"%H")
minute=$(date +"%M")
TIME="${month}${day}${hour}${minute}"

# Extract algo config name for output dir
algo_name=$(basename "$algo_cfg_path" .json)

# ======================== Run ========================
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    echo "=============================================="
    echo "Sequence length: ${MAX_SEQ_LENGTH}"
    echo "=============================================="

    # Directories
    ROOT_DIR="${BENCHMARK_DIR}/results_ruler/${MODEL_NAME}/${algo_name}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${ROOT_DIR}/data"
    PRED_DIR="${ROOT_DIR}/pred"
    mkdir -p "${DATA_DIR}" "${PRED_DIR}"

    # Step 1: Generate RULER data (using RULER's prepare.py)
    for TASK in "${TASKS[@]}"; do
        TASK_DATA="${DATA_DIR}/${TASK}/validation.jsonl"
        if [ -f "${TASK_DATA}" ]; then
            echo "[Data] ${TASK} already exists at seq_len=${MAX_SEQ_LENGTH}, skipping."
        else
            echo "[Data] Generating ${TASK} at seq_len=${MAX_SEQ_LENGTH}..."
            python "${RULER_DIR}/data/prepare.py" \
                --save_dir "${DATA_DIR}" \
                --benchmark synthetic \
                --task "${TASK}" \
                --tokenizer_path "${MODEL_PATH}" \
                --tokenizer_type "${TOKENIZER_TYPE}" \
                --max_seq_length "${MAX_SEQ_LENGTH}" \
                --model_template_type "${MODEL_TEMPLATE_TYPE}" \
                --num_samples "${NUM_SAMPLES}"
        fi
    done

    # Step 2: Run prediction with sparse attention
    for TASK in "${TASKS[@]}"; do
        PRED_FILE="${PRED_DIR}/${TASK}.jsonl"
        if [ -f "${PRED_FILE}" ]; then
            echo "[Pred] ${TASK} already exists at seq_len=${MAX_SEQ_LENGTH}, skipping."
            continue
        fi

        BASE_TASK="${TASK_TO_BASE[$TASK]}"
        MAX_GEN="${TASK_MAX_GEN[$BASE_TASK]}"

        echo "[Pred] Running ${TASK} at seq_len=${MAX_SEQ_LENGTH} (max_gen=${MAX_GEN})..."
        CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "${BENCHMARK_DIR}/RULER/pred.py" \
            --model_name "${MODEL_NAME}" \
            --model_path "${MODEL_PATH}" \
            --algo-config-path "${algo_cfg_path}" \
            --data_dir "${DATA_DIR}" \
            --save_dir "${PRED_DIR}" \
            --task "${TASK}" \
            --max_gen "${MAX_GEN}" \
            --subset validation
    done

    # Step 3: Evaluate (using our own evaluate.py, no nemo dependency)
    echo "[Eval] Evaluating seq_len=${MAX_SEQ_LENGTH}..."
    python "${BENCHMARK_DIR}/RULER/evaluate.py" \
        --data_dir "${PRED_DIR}"

    echo "Results saved to ${PRED_DIR}/summary.csv"
done

echo ""
echo "=============================================="
echo "All done! Time: ${TIME}"
echo "Results root: ${BENCHMARK_DIR}/results_ruler/${MODEL_NAME}/${algo_name}/"
echo "=============================================="
