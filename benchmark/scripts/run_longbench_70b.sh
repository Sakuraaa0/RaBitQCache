#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_longbench_70b.sh <algo_config_path>
# Example: CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_longbench_70b.sh configs/configs_rabitq/rabitq_0.85.json
#
# Requires 2x H200 GPUs. The model is loaded with device_map="auto" which
# automatically splits the 70B model across available GPUs.

MODEL=Meta-Llama-3.1-70B-Instruct
MODELPATH=/mnt/user-ssd/your_user/rabitq/models/Llama-3.1-70B-Instruct  # Adjust to your local path

OUTPUT_DIR=results_longbench/$MODEL

# Use HF load_dataset. Modify this if you want to use local data set.
DATASET_PATH="./LongBench/data/data"

CONFIG_PATH=LongBench/config/

mkdir -p $OUTPUT_DIR

algo_cfg_path=$1
if [ -z "$algo_cfg_path" ]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_longbench_70b.sh <algo_config_path>"
    exit 1
fi

# Whether enable LongBench-E benchmark
enable_e=0

datasets=('narrativeqa' 'qasper' 'multifieldqa_en' 'hotpotqa'
          '2wikimqa' 'musique' 'gov_report' 'qmsum' 'multi_news'
          'triviaqa' 'passage_retrieval_en' 'lcc' 'repobench-p')

dataset_e=('qasper' 'multifieldqa_en' 'hotpotqa' '2wikimqa' 'gov_report'
           'multi_news' 'trec' 'triviaqa' 'passage_count'
           'passage_retrieval_en' 'lcc' 'repobench-p')

if [ "$enable_e" -eq 0 ]; then
  benchmark_dataset=("${datasets[@]}")
else
  benchmark_dataset=("${dataset_e[@]}")
fi

month=$(date +"%m")
day=$(date +"%d")
hour=$(date +"%H")
minute=$(date +"%M")
time="${month}${day}${hour}${minute}"

for task in "${benchmark_dataset[@]}"
do
    echo "Running $task:"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u LongBench/pred.py \
        --model $MODEL --model_path $MODELPATH --task $task \
        --algo-config-path $algo_cfg_path \
        --e $enable_e --t $time \
        --dataset-path $DATASET_PATH \
        --output-dir $OUTPUT_DIR \
        --config-path $CONFIG_PATH
done
