#!/bin/bash

# Arrays for multiple model bases and configs
MODEL_BASES=(
    "/home/aleksandar/mmdetsatmae"
    "/home/aleksandar/mmdetscalemae"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae.py"
)

AL_DIR="/home/aleksandar/activeLearningIDS"
DEVICE=0  # For evaluation
PERCENTAGES=(1 3 5 10)

# Loop through model bases and config files
for i in "${!MODEL_BASES[@]}"; do
    MODEL_BASE="${MODEL_BASES[$i]}"
    CONFIG_BASE="${CONFIG_BASES[$i]}"

    echo "============================"
    echo "Running for Model: $MODEL_BASE"
    echo "Using Config: $CONFIG_BASE"
    echo "============================"

    for X in "${PERCENTAGES[@]}"; do
        RANDOM_FILE="$AL_DIR/random_${X}.csv"
        SELECTED_FILE="$AL_DIR/selectedIDS.csv"
        WORK_DIR="${MODEL_BASE}_${X}"

        # 1. Copy current random_X.csv to selectedIDS.csv
        cp "$RANDOM_FILE" "$SELECTED_FILE"
        echo "Copied $RANDOM_FILE to $SELECTED_FILE"

        # 2. Train model (use proper GPU)
        CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29619 tools/train.py "$CONFIG_BASE" --work-dir "$WORK_DIR"

        # 3. Evaluate all epoch checkpoints
        CKPT="$WORK_DIR"/epoch_12.pth
        EPOCH_NUM=$(echo "$CKPT" | grep -oP 'epoch_\K[0-9]+')
        OUT_FILE="$WORK_DIR/results_epoch_${EPOCH_NUM}.pkl"
        echo "Evaluating checkpoint: $CKPT -> $OUT_FILE"
        CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE" "$CKPT" --out "$OUT_FILE"

    done
done
