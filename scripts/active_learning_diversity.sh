#!/bin/bash

MODEL_BASES=(
    "/home/aleksandar/faster_rcnn_rvsa_l_800_mae_mtp_dior"
    "/home/aleksandar/mmdetdino784LowLR"
    "/home/aleksandar/mmdetdinodinohead784"
    "/home/aleksandar/mmdetsatmae"
    "/home/aleksandar/mmdetscalemae"
    "/home/aleksandar/mmdetdinolarge784"
    "/home/aleksandar/mmdetdinodinoheadlarge"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtpdinohead_copy.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtpdinohead_copylarge.py"
)

CONFIG_BASES_VAL=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior_unc_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtpdinohead_copy_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge_train_val.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtpdinohead_copylarge_train_val.py"
)

IDS_CHECKPOINT=(
    "/home/aleksandar/activeLearningIDS/mtp"
    "/home/aleksandar/activeLearningIDS/dinov2"
    "/home/aleksandar/activeLearningIDS/dinov2dinohead"
    "/home/aleksandar/activeLearningIDS/satmae"
    "/home/aleksandar/activeLearningIDS/scalemae"
    "/home/aleksandar/activeLearningIDS/dinov2large"
    "/home/aleksandar/activeLearningIDS/dinov2dinoheadlarge"
)

# Active learning IDs dir
AL_DIR="/home/aleksandar/activeLearningIDS"
# GPU device
DEVICE=0

# Overall percentages for splits you want to end up with
PERCENTAGES=(1 3 5 10 0)
# Per-round ADD percentages (from current to next target)
# e.g., 1% -> +2% = 3%; 3% -> +2% = 5%; 5% -> +5% = 10%
ADD_PERCENTS=(2 5 0)

# Diversity config
DIVERSITY_DISTANCE="cosine"   # cosine or euclidean
NUM_CLASSES=20                # for class histogram features

# Selection mode for this script:
#   PURE diversity:  DIVERSITY_MODE="pure"
#   HYBRID (uncertainty prefilter + clustering): DIVERSITY_MODE="hybrid"
DIVERSITY_MODE="pure"
POOL_PERCENT=10               # for hybrid: prefilter top 10% most uncertain

# If hybrid, we need an uncertainty ranking; configure method here
UNC_METHOD="average"          # leastconf | average | count (your script supports these)
UNC_RANK_CSV="$AL_DIR/uncertainty_ranked.csv"



############################################
# Main loop
############################################
for i in "${!MODEL_BASES[@]}"; do
  MODEL_BASE="${MODEL_BASES[$i]}"
  CONFIG_BASE="${CONFIG_BASES[$i]}"
  CONFIG_BASE_VAL="${CONFIG_BASES_VAL[$i]}"
  CHECKPOINT_DIR="${IDS_CHECKPOINT[$i]}"

  echo "==========================================="
  echo "Model:     $MODEL_BASE"
  echo "Config:    $CONFIG_BASE"
  echo "Val cfg:   $CONFIG_BASE_VAL"
  echo "Mode:      diversity ($DIVERSITY_MODE)"
  echo "==========================================="

  mkdir -p "$CHECKPOINT_DIR"

  # 0) Seed with the same 1% as your random baseline
  cp "$AL_DIR/random_1.csv" "$AL_DIR/div_1.csv"
  cp "$AL_DIR/div_1.csv" "$AL_DIR/selectedIDS.csv"

  # Create working dir for the 1% (assuming it was already trained as RANDOM_AL_WORK_DIR)
  RANDOM_AL_WORK_DIR="${MODEL_BASE}_1"
  WORK_DIR="${MODEL_BASE}_div_${DIVERSITY_MODE}_1"
  mkdir -p "$WORK_DIR"

  # Copy initial checkpoint & pkl results if you already trained the 1% baseline
  if [[ -f "$RANDOM_AL_WORK_DIR/epoch_12.pth" ]]; then
    cp "$RANDOM_AL_WORK_DIR/epoch_12.pth" "$WORK_DIR/epoch_12.pth"
  fi
  if [[ -f "$RANDOM_AL_WORK_DIR/results_epoch_12.pkl" ]]; then
    cp "$RANDOM_AL_WORK_DIR/results_epoch_12.pkl" "$WORK_DIR/results_epoch_12.pkl"
  fi

  # Validate on pool for diversity (and possibly for uncertainty prefilter)
  CKPT="$WORK_DIR/epoch_12.pth"
  VAL_OUT="$WORK_DIR/results_epoch_12_div.pkl"
  CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE_VAL" "$CKPT" --out "$VAL_OUT"
  


  # First addition: from 1% -> 3% (ADD_PERCENTS[0])
  ADD_FIRST="${ADD_PERCENTS[0]}"
  
  python tools/get_dataset_diversity.py \
    --model-path "$VAL_OUT" \
    --output-csv "$AL_DIR/diverse_to_add.csv" \
    --n-to-add "$2" \
    --num-classes "$NUM_CLASSES" \
    --distance "$DIVERSITY_DISTANCE"

 
  # Create div_3.csv
  PREV_IDS="$AL_DIR/div_1.csv"
  NEXT_IDS="$AL_DIR/div_3.csv"
  # 5. Compute uncertainty and make new splits (repeat for 3, 5, 10%)
        # Save header
    head -n 1 "$PREV_IDS" > "$NEXT_IDS"
    # Save all data, skipping headers, sort, uniq, and append
    (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/diverse_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"

  # Now iterate the remaining rounds: 3%->5%, 5%->10%
  for round in 0 1 2; do
    NEXT_PERC="${PERCENTAGES[$((round+2))]}"
    CURRENT_PERC="${PERCENTAGES[$((round+1))]}"
    ADD_PERC="${ADD_PERCENTS[$round]}"

    PREV_IDS="$AL_DIR/div_${CURRENT_PERC}.csv"
    NEXT_IDS="$AL_DIR/div_${NEXT_PERC}.csv"

    WORK_DIR_NEXT="${MODEL_BASE}_div_${DIVERSITY_MODE}_${CURRENT_PERC}"
    CKPT_NEXT="$WORK_DIR_NEXT/epoch_12.pth"
    VAL_OUT_NEXT="$WORK_DIR_NEXT/results_epoch_12_div.pkl"
    TEST_OUT_NEXT="$WORK_DIR_NEXT/results_epoch_12.pkl"

    echo "---- Round $round: ${CURRENT_PERC}% -> ${NEXT_PERC}% (+${ADD_PERC}%) ----"
    # Set selected IDs for training
    cp "$PREV_IDS" "$AL_DIR/selectedIDS.csv"

    # Train
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun --master_port 29619 tools/train.py "$CONFIG_BASE" --work-dir "$WORK_DIR_NEXT"

    # Validation (for selection)
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE_VAL" "$CKPT_NEXT" --out "$VAL_OUT_NEXT"

    # (Optional) test on train cfg for bookkeeping
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE" "$CKPT_NEXT" --out "$TEST_OUT_NEXT"

   

      python tools/get_dataset_diversity.py \
        --model-path "$VAL_OUT_NEXT" \
        --output-csv "$AL_DIR/diverse_to_add.csv" \
        --n-to-add "$ADD_PERC" \
        --num-classes "$NUM_CLASSES" \
        --distance "$DIVERSITY_DISTANCE"


    # Merge to create next split (div_5.csv then div_10.csv)
    head -n 1 "$PREV_IDS" > "$NEXT_IDS"
        # Save all data, skipping headers, sort, uniq, and append
        (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/diverse_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
  done

  # Save final splits for this model
  mkdir -p "$CHECKPOINT_DIR"
  cp "$AL_DIR/div_1.csv"  "$CHECKPOINT_DIR/div_1.csv"
  cp "$AL_DIR/div_3.csv"  "$CHECKPOINT_DIR/div_3.csv"
  cp "$AL_DIR/div_5.csv"  "$CHECKPOINT_DIR/div_5.csv"
  cp "$AL_DIR/div_10.csv" "$CHECKPOINT_DIR/div_10.csv"
  echo "Copied diversity split csvs to $CHECKPOINT_DIR"

done