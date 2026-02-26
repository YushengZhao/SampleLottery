#!/bin/bash

cleanup() {
    echo "============ Script exiting, cleaning up background processes... ============"
    pkill -P $$
    echo "============ Cleanup complete. ============"
}

trap cleanup EXIT INT TERM

set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MAX_TOKENS_PER_CALL=$3
OUTPUT_DIR=$4

SPLIT="test"
NUM_TEST_SAMPLE=-1

ALL_DATA_NAMES="amc23x8,minerva_math,olympiadbench,math500"

IFS=',' read -ra DATASETS <<< "$ALL_DATA_NAMES"
ALL_EXIST=true
for DATASET in "${DATASETS[@]}"; do
    if [ ! -d "${OUTPUT_DIR}/${DATASET}" ] || [ -z "$(find ${OUTPUT_DIR}/${DATASET} -name '*metrics.json' -print -quit)" ]; then
        ALL_EXIST=false
        break
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo "============ All datasets in ${ALL_DATA_NAMES} have been evaluated. Skipping. ============="
    if [ -f "${OUTPUT_DIR}/summary_metrics.json" ]; then
        echo "============ Existing Summary ============="
        cat "${OUTPUT_DIR}/summary_metrics.json"
    fi
else
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${ALL_DATA_NAMES} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0.6 \
        --n_sampling 32 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call ${MAX_TOKENS_PER_CALL}
fi