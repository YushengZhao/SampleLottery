PROMPT_TYPE="llama"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MAX_TOKENS="3072"

CHECKPOINTS_DIR="../checkpoints"

export WANDB_MODE=offline

PROJECT_NAME="al"
EXPERIMENT_NAME="LLaMA-3.1-8B-Instruct-CONST_k4"
GLOBAL_STEP_LIST=($(seq 20 20 60))

LOG_FILE="${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/eval/evaluation_log.txt"
mkdir -p $(dirname ${LOG_FILE})
echo "Logging evaluation output to ${LOG_FILE}"

for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
    echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
    MODEL_NAME_OR_PATH=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}/actor
    OUTPUT_DIR=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/eval/global_step_${GLOBAL_STEP}
    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR 2>&1 | tee -a ${LOG_FILE}
done
