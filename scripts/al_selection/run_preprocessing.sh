#!/bin/bash
set -e
CONFIG_FILE="al/config.py"

if [ -z "$1" ]; then
    echo "Error: Model path must be provided as the first argument."
    echo "Usage: ./scripts/al_selection/run_preprocessing.sh /path/to/model"
    exit 1
fi
MODEL_PATH=$1
MODEL_NAME=$(basename "$MODEL_PATH")
echo "Starting preprocessing for model: $MODEL_NAME"

cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
trap 'mv "$CONFIG_FILE.bak" "$CONFIG_FILE"' EXIT

sed -i "s#^    base_model_path: str = .*#    base_model_path: str = \"$MODEL_PATH\"#" "$CONFIG_FILE"

echo "--> Running 00_embed_problems.py"
python al/00_embed_problems.py

echo "--> Running 01_calibrate.py"
python al/01_calibrate.py

echo "--> Running 02_screen_pool.py"
python al/02_screen_pool.py

echo "Preprocessing complete for model: $MODEL_NAME"