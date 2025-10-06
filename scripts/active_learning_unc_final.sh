#!/bin/bash

MODEL_BASES=(
    "/home/aleksandar/faster_rcnn_rvsa_l_800_mae_mtp_dior"
    "/home/aleksandar/mmdetdino784LowLR"
    "/home/aleksandar/mmdetsatmae"
    "/home/aleksandar/mmdetscalemae"
    "/home/aleksandar/mmdetdinolarge784"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge.py"
)

CONFIG_BASES_VAL=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior_unc_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge_train_val.py"
)

IDS_CHECKPOINT=(
    "/home/aleksandar/activeLearningIDS/mtp"
    "/home/aleksandar/activeLearningIDS/dinov2"
    "/home/aleksandar/activeLearningIDS/satmae"
    "/home/aleksandar/activeLearningIDS/scalemae"
    "/home/aleksandar/activeLearningIDS/dinov2large"
)


AL_DIR="/home/aleksandar/activeLearningIDS"
DEVICE=2
PERCENTAGES=(10 20 40 60 80 0)
UNC_PERCENTS=(20 20 20 0)  # how much to add each round: 3→5 (add 2%), 5→10 (add 5%), 10 is last
UNC_METHOD="count" # choose between margin or leastconf
for i in "${!MODEL_BASES[@]}"; do
    MODEL_BASE="${MODEL_BASES[$i]}"
    CONFIG_BASE="${CONFIG_BASES[$i]}"
    CONFIG_BASE_VAL="${CONFIG_BASES_VAL[$i]}"
    echo "============================"
    echo "Running for Model: $MODEL_BASE"
    echo "Using Config: $CONFIG_BASE"
    echo "============================"

    # 1. Copy unc_1.csv as starting point
    cp "$AL_DIR/random_10.csv" "$AL_DIR/unc_10.csv"
    cp "$AL_DIR/unc_10.csv" "$AL_DIR/selectedIDS.csv"
    WORK_DIR="${MODEL_BASE}_unc_${UNC_METHOD}_10"
    RANDOM_AL_WORK_DIR="${MODEL_BASE}_10"
    mkdir "$WORK_DIR"
    cp "$RANDOM_AL_WORK_DIR/epoch_12.pth" "$WORK_DIR/epoch_12.pth"
    # 2. (Optional) If you want to train the first model again, do it here, or comment if already done

    # 3. Run "validation" for uncertainty on pool
    CKPT="$WORK_DIR/epoch_12.pth"
    VAL_OUT="$WORK_DIR/results_epoch_12_unc.pkl"
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE_VAL" "$CKPT" --out "$VAL_OUT"

    cp "$RANDOM_AL_WORK_DIR/results_epoch_12.pkl" "$WORK_DIR/results_epoch_12.pkl"
    # Uncertainty selection
    python tools/get_dataset_uncertainty.py \
        --uncertainty-csv "$AL_DIR/uncertainty_ranked.csv" \
        --model-path "$VAL_OUT" \
        --n-to-add "10" \
        --output-csv "$AL_DIR/uncertain_to_add.csv"\
        --uncertainty-method "$UNC_METHOD"
    PREV_IDS="$AL_DIR/unc_10.csv"
    NEXT_IDS="$AL_DIR/unc_20.csv"
    # 5. Compute uncertainty and make new splits (repeat for 3, 5, 10%)
        # Save header
    head -n 1 "$PREV_IDS" > "$NEXT_IDS"
    # Save all data, skipping headers, sort, uniq, and append
    (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
    for round in 0 1 2 3; do
        NEXT_PERC="${PERCENTAGES[$((round+2))]}"
        CURRENT_PERC="${PERCENTAGES[$((round+1))]}"
        PREV_IDS="$AL_DIR/unc_${PERCENTAGES[$((round+1))]}.csv"
        NEXT_IDS="$AL_DIR/unc_${NEXT_PERC}.csv"
        WORK_DIR_NEXT="${MODEL_BASE}_unc_${UNC_METHOD}_${CURRENT_PERC}"
        CKPT_NEXT="$WORK_DIR_NEXT/epoch_12.pth"
        VAL_OUT_NEXT="$WORK_DIR_NEXT/results_epoch_12_unc.pkl"
        TEST_OUT_NEXT="$WORK_DIR_NEXT/results_epoch_12.pkl"
        echo "Now working on round $round with Current ID=$PREV_IDS, NEXT_IDS=$NEXT_IDS"
        # Train next round
        cp "$PREV_IDS" "$AL_DIR/selectedIDS.csv"
        CUDA_VISIBLE_DEVICES=$DEVICE torchrun --master_port 29619 tools/train.py "$CONFIG_BASE" --work-dir "$WORK_DIR_NEXT"

        # Validation
        CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE_VAL" "$CKPT_NEXT" --out "$VAL_OUT_NEXT"

        # Test
        CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE" "$CKPT_NEXT" --out "$TEST_OUT_NEXT"

        # Uncertainty selection
        python tools/get_dataset_uncertainty.py \
            --uncertainty-csv "$AL_DIR/uncertainty_ranked.csv" \
            --model-path "$VAL_OUT_NEXT" \
            --n-to-add "20" \
            --output-csv "$AL_DIR/uncertain_to_add.csv" \
            --uncertainty-method "$UNC_METHOD"

        # Make next split
        echo "Now adding to $PREV_IDS, and creating $NEXT_IDS"
        # Save header
        head -n 1 "$PREV_IDS" > "$NEXT_IDS"
        # Save all data, skipping headers, sort, uniq, and append
        (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
    done

      CHECKPOINT_DIR="${IDS_CHECKPOINT[$i]}"
    mkdir -p "$CHECKPOINT_DIR"
    cp "$AL_DIR/unc_20.csv" "$CHECKPOINT_DIR/unc_20.csv"
    cp "$AL_DIR/unc_40.csv" "$CHECKPOINT_DIR/unc_40.csv"
    cp "$AL_DIR/unc_60.csv" "$CHECKPOINT_DIR/unc_60.csv"
    cp "$AL_DIR/unc_80.csv" "$CHECKPOINT_DIR/unc_80.csv"
    echo "Copied split csvs to $CHECKPOINT_DIR"
done