#!/bin/bash
set -e
CONFIG_FILE="al/config.py"
BUDGETS=(4 8)
PAD_TO_SIZE=64

if [ -z "$1" ]; then
    echo "Error: Model path must be provided as the first argument."
    echo "Usage: ./scripts/al_selection/run_robustness_study.sh /path/to/model"
    exit 1
fi
MODEL_PATH=$1
MODEL_NAME=$(basename "$MODEL_PATH")

echo "================================================================="
echo "Starting Robustness Study for model: $MODEL_NAME"
echo "(Calibrating on MMLU and applying to BIG-MATH)"
echo "================================================================="

cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
trap 'mv "$CONFIG_FILE.bak" "$CONFIG_FILE"; echo "Restored original config.py."' EXIT

sed -i "s#^    base_model_path: str = .*#    base_model_path: str = \"$MODEL_PATH\"#" "$CONFIG_FILE"
echo "Model path in config.py updated for this session."
echo "-----------------------------------------------------------------"

echo "--> Part 1: Running Robustness Analysis (Calibrate on MMLU, Rescore BIG-MATH)"
python -m al.CONST.further_studies.run_robustness_analysis
echo "--> Part 1 Complete. MMLU-calibrated scores are now available."
echo "-----------------------------------------------------------------"

echo "--> Part 2: Generating CONST training sets using MMLU-calibrated scores"
for budget in "${BUDGETS[@]}"; do
    echo "--> Generating training set for CONST-MMLU with budget=${budget}"

    python -m al.CONST.select_samples \
        --num_clusters "$budget" \
        --top_n_per_cluster 1 \
        --pad_to_size "$PAD_TO_SIZE" \
        --screened_results_file "further_studies/screened_results_mmlu_calibrated.jsonl" \
        --output_subdir "FurtherAnalysis" \
        --output_prefix "CONST_k8_CalibMMLU"
done

echo "--> Part 2 Complete."
echo "-----------------------------------------------------------------"
echo "Robustness study has been successfully completed."
echo "Datasets are located in: data/train/$MODEL_NAME/FurtherAnalysis/"