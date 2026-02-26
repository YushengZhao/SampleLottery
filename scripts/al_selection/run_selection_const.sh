#!/bin/bash
set -e
CONFIG_FILE="al/config.py"
BUDGETS=(4 8)
PAD_TO_SIZE=64

if [ -z "$1" ]; then
    echo "Error: Model path must be provided." >&2
    exit 1
fi
MODEL_PATH=$1
echo "Starting CONST selection for model: $(basename "$MODEL_PATH")"

cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
trap 'mv "$CONFIG_FILE.bak" "$CONFIG_FILE"' EXIT
sed -i "s#^    base_model_path: str = .*#    base_model_path: str = \"$MODEL_PATH\"#" "$CONFIG_FILE"

for budget in "${BUDGETS[@]}"; do
    echo "--> Generating training set for CONST with budget=${budget}"
    python -m al.CONST.select_samples \
        --num_clusters "$budget" \
        --top_n_per_cluster 1 \
        --pad_to_size "$PAD_TO_SIZE"
done
echo "CONST selection complete."