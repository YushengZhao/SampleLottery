#!/bin/bash
set -e
CONFIG_FILE="al/config.py"
PAD_TO_SIZE=64
NUM_SAMPLES=8

if [ -z "$1" ]; then
    echo "Error: Model path must be provided as the first argument."
    echo "Usage: ./scripts/al_selection/run_ablations_const.sh /path/to/model"
    exit 1
fi
MODEL_PATH=$1
MODEL_NAME=$(basename "$MODEL_PATH")
echo "============================================================"
echo "Starting CONST Ablation Studies for model: $MODEL_NAME"
echo "============================================================"

cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
trap 'mv "$CONFIG_FILE.bak" "$CONFIG_FILE"; echo "Restored original config.py."' EXIT

sed -i "s#^    base_model_path: str = .*#    base_model_path: str = \"$MODEL_PATH\"#" "$CONFIG_FILE"
echo "Model path in config.py updated for this session."
echo "------------------------------------------------------------"

echo "--> Running Ablation V1: Cluster and Randomly Select"
python -m al.CONST.ablations.select_v1_random \
    --num_samples "$NUM_SAMPLES" \
    --pad_to_size "$PAD_TO_SIZE"
echo "--> V1 Complete."
echo "------------------------------------------------------------"

echo "--> Running Ablation V2: Top-K AL Score (No Clustering)"
python -m al.CONST.ablations.select_v2_no_cluster \
    --num_samples "$NUM_SAMPLES" \
    --pad_to_size "$PAD_TO_SIZE"
echo "--> V2 Complete."
echo "------------------------------------------------------------"

echo "--> Running Ablation V3: Cluster and Select Highest Entropy"
python -m al.CONST.ablations.select_v3_entropy \
    --num_samples "$NUM_SAMPLES" \
    --pad_to_size "$PAD_TO_SIZE"
echo "--> V3 Complete."
echo "------------------------------------------------------------"

echo "--> Running Ablation V4: Cluster (k=4) and Select Top-2 per Cluster"
python -m al.CONST.ablations.select_v4_cluster_config \
    --num_clusters_v4 4 \
    --top_n_per_cluster_v4 2 \
    --pad_to_size "$PAD_TO_SIZE"
echo "--> V4 Complete."
echo "------------------------------------------------------------"

echo "--> Running Ablation V5: Outcome Volatility Only"
python -m al.CONST.ablations.select_v5_outcome_only \
    --num_samples "$NUM_SAMPLES" \
    --pad_to_size "$PAD_TO_SIZE"
echo "--> V5 Complete."
echo "------------------------------------------------------------"

echo "--> Running Ablation V6: Procedural Volatility Only"
python -m al.CONST.ablations.select_v6_procedural_only \
    --num_samples "$NUM_SAMPLES" \
    --pad_to_size "$PAD_TO_SIZE"
echo "--> V6 Complete."
echo "------------------------------------------------------------"

echo "All CONST ablation studies have been successfully completed."
echo "Datasets are located in: data/train/$MODEL_NAME/AblationStudy/"