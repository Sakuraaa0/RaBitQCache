#!/bin/bash
# Standalone evaluation for RULER predictions.
# Usage: bash scripts/eval_ruler.sh <pred_dir>
# Example: bash scripts/eval_ruler.sh results_ruler/Meta-Llama-3.1-8B-Instruct/rabitq_0.65/4096/pred
#
# Or evaluate all seq lengths for one config:
# bash scripts/eval_ruler.sh results_ruler/Meta-Llama-3.1-8B-Instruct/rabitq_0.65

set -e

RULER_DIR=/mnt/user-ssd/your_user/rabitq/RULER/scripts

if [ -z "$1" ]; then
    echo "Usage: bash scripts/eval_ruler.sh <pred_dir_or_config_dir>"
    exit 1
fi

TARGET=$1

# Check if it's a single pred dir (contains .jsonl files) or a config dir (contains seq_length subdirs)
if ls "${TARGET}"/*.jsonl &>/dev/null 2>&1; then
    # Single pred directory
    echo "Evaluating: ${TARGET}"
    cd "${RULER_DIR}"
    python eval/evaluate.py --data_dir "${TARGET}" --benchmark synthetic
    echo "Results: ${TARGET}/summary.csv"
else
    # Config directory with seq_length subdirs
    for SEQ_DIR in "${TARGET}"/*/; do
        PRED_DIR="${SEQ_DIR}pred"
        if [ -d "${PRED_DIR}" ]; then
            SEQ_LEN=$(basename "${SEQ_DIR}")
            echo "=============================================="
            echo "Evaluating seq_len=${SEQ_LEN}"
            echo "=============================================="
            cd "${RULER_DIR}"
            python eval/evaluate.py --data_dir "${PRED_DIR}" --benchmark synthetic
            echo ""
        fi
    done
fi
