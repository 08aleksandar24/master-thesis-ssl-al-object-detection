#!/bin/bash

# Arrays for multiple model bases and configs
MODEL_BASES=(
    "/home/aleksandar/faster_rcnn_rvsa_l_800_mae_mtp_dior"
    "/home/aleksandar/mmdetsatmae"
    "/home/aleksandar/mmdetscalemae"
    "/home/aleksandar/mmdetdinolarge784"
    "/home/aleksandar/mmdetdinov3"
    "/home/aleksandar/mmdetdino784LowLR"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp.py"
)

CONFIG_BASES_VAL=(
  "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior_unc_train_val.py"
  "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae_train_val.py"
  "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae_train_val.py"
  "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge_train_val.py"
  "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3_train_val.py"
  "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp_train_val.py"
)

AL_DIR="/home/aleksandar/activeLearningIDS"
DEVICE=3  # For evaluation
# Selection params for jump 10% -> 20%
ADD_TO_20=10

# Uncertainty methods to compute
UNC_METHODS=( "leastconf" "average" "count" )

# Diversity settings
DIVERSITY_DISTANCE="cosine"  # cosine|euclidean
NUM_CLASSES=20
DIVERSITY_MODE="pure"        # only used for labeling; we run diversity scorer directly
PREV_IDS="/home/aleksandar/activeLearningIDS/random_10.csv"
#cp "/home/aleksandar/activeLearningIDS/random_20.csv" "/home/aleksandar/activeLearningIDS/selectedIDS.csv"
#for i in "${!MODEL_BASES[@]}"; do
 # MODEL_BASE="${MODEL_BASES[$i]}"
 # CONFIG_BASE="${CONFIG_BASES[$i]}"
 # WORK_DIR_20="${MODEL_BASE}_20"
#  CKPT_20="${WORK_DIR_20}/epoch_12.pth"
#  OUT_20="${WORK_DIR_20}/results_epoch_12.pkl"
#  CUDA_VISIBLE_DEVICES=$DEVICE torchrun --master_port 29619 tools/train.py "$CONFIG_BASE" --work-dir "$WORK_DIR_20"
#  if exists "$CKPT_20"; then
#    log "Testing RANDOM_20 for: $MODEL_BASE -> $OUT_20"
#    CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE" "$CKPT_20" --out "$OUT_20"
#  else
#    log "[SKIP] No RANDOM_20 checkpoint: $CKPT_20"
#  fi
#done

log "=== PART B: Derive 20% IDs from 10% checkpoints and test 10% models (no training) ==="
for i in "${!MODEL_BASES[@]}"; do
  MODEL_BASE="${MODEL_BASES[$i]}"
  CFG="${CONFIG_BASES[$i]}"
  CFG_VAL="${CONFIG_BASES_VAL[$i]}"

  WORK_DIR_10="${MODEL_BASE}_10"
  CKPT_10="${WORK_DIR_10}/epoch_12.pth"
  TEST_OUT_10="${WORK_DIR_10}/results_epoch_12.pkl"
  VAL_OUT_10="${WORK_DIR_10}/results_epoch_12_unc.pkl"

  # Ensure 10% checkpoint exists
  if ! exists "$CKPT_10"; then
    log "[trace SKIP] No 10% checkpoint for $MODEL_BASE: $CKPT_10"

  fi

  # 1) Validation on the pool (to score remaining samples)
  log "[VAL] Pool scoring @10% -> $VAL_OUT_10"
  CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CFG_VAL" "$CKPT_10" --out "$VAL_OUT_10"

  # 2) UNCERTAINTY: compute IDs to reach 20% and create *_20.csv for each method
  for M in "${UNC_METHODS[@]}"; do
   
    WORK_DIR_20="${MODEL_BASE}_unc_${M}_20"
    CKPT_20="${WORK_DIR_20}/epoch_12.pth"
    TEST_OUT_20="${WORK_DIR_20}/results_epoch_12.pkl"

    # 2.a rank to-add from pool
    TO_ADD_CSV="$AL_DIR/uncertain_to_add.csv"
    python tools/get_dataset_uncertainty.py \
      --uncertainty-csv "$AL_DIR/uncertainty_ranked.csv" \
      --model-path "$VAL_OUT_10" \
      --n-to-add "$ADD_TO_20" \
      --output-csv "$TO_ADD_CSV" \
      --uncertainty-method "$M"

    # 2.b merge with 10% to make 20%
    NEXT_IDS="/home/aleksandar/activeLearningIDS/unc_20.csv"
    head -n 1 "$PREV_IDS" > "$NEXT_IDS"
    # Save all data, skipping headers, sort, uniq, and append
    (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"
    cp "$NEXT_IDS" "/home/aleksandar/activeLearningIDS/selectedIDS.csv"
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun --master_port 29619 tools/train.py "$CFG" --work-dir "$WORK_DIR_20"
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CFG" "$CKPT_20" --out "$TEST_OUT_20"
  done

  # 3) DIVERSITY: compute IDs to reach 20% and create div_20.csv
    WORK_DIR_20="${MODEL_BASE}_div_pure_20"
    CKPT_20="${WORK_DIR_20}/epoch_12.pth"
    TEST_OUT_20="${WORK_DIR_20}/results_epoch_12.pkl"

    DIV_TO_ADD="$AL_DIR/diverse_to_add.csv"
    python tools/get_dataset_diversity.py \
        --model-path "$VAL_OUT_10" \
        --output-csv "$DIV_TO_ADD" \
        --n-to-add "$ADD_TO_20" \
        --num-classes "$NUM_CLASSES" \
        --distance "$DIVERSITY_DISTANCE"

    NEXT_IDS="/home/aleksandar/activeLearningIDS/div_20.csv"
    head -n 1 "$PREV_IDS" > "$NEXT_IDS"
    # Save all data, skipping headers, sort, uniq, and append
    (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/diverse_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"
    cp "$NEXT_IDS" "/home/aleksandar/activeLearningIDS/selectedIDS.csv"
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun --master_port 29619 tools/train.py "$CFG" --work-dir "$WORK_DIR_20"
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CFG" "$CKPT_20" --out "$TEST_OUT_20"
   


done

log "=== Done. Random_20 tests + 20% ID generation from 10% ckpts complete. ==="