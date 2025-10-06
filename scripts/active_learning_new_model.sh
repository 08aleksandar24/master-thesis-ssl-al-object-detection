#!/bin/bash

# Arrays for multiple model bases and configs
MODEL_BASES=(
    "/home/aleksandar/mmdetdinov3"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py"
)

AL_DIR="/home/aleksandar/activeLearningIDS"
DEVICE=3  # For evaluation
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
        CUDA_VISIBLE_DEVICES=$DEVICE torchrun --master_port 29619 tools/train.py "$CONFIG_BASE" --work-dir "$WORK_DIR"

        # 3. Evaluate all epoch checkpoints
        CKPT="$WORK_DIR"/epoch_12.pth
        EPOCH_NUM=$(echo "$CKPT" | grep -oP 'epoch_\K[0-9]+')
        OUT_FILE="$WORK_DIR/results_epoch_${EPOCH_NUM}.pkl"
        echo "Evaluating checkpoint: $CKPT -> $OUT_FILE"
        CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE" "$CKPT" --out "$OUT_FILE"

    done
done


MODEL_BASES=(
    "/home/aleksandar/mmdetdinov3"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py"
)

CONFIG_BASES_VAL=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3_train_val.py"
)

IDS_CHECKPOINT=(
    "/home/aleksandar/activeLearningIDS/dinov3"
)

AL_DIR="/home/aleksandar/activeLearningIDS"
DEVICE=3
PERCENTAGES=(1 3 5 10 0)
UNC_PERCENTS=(2 5 0)  # how much to add each round: 3→5 (add 2%), 5→10 (add 5%), 10 is last
UNC_METHOD="average" # choose between margin or leastconf
for i in "${!MODEL_BASES[@]}"; do
    MODEL_BASE="${MODEL_BASES[$i]}"
    CONFIG_BASE="${CONFIG_BASES[$i]}"
    CONFIG_BASE_VAL="${CONFIG_BASES_VAL[$i]}"
    echo "============================"
    echo "Running for Model: $MODEL_BASE"
    echo "Using Config: $CONFIG_BASE"
    echo "============================"

    # 1. Copy unc_1.csv as starting point
    cp "$AL_DIR/random_1.csv" "$AL_DIR/unc_1.csv"
    cp "$AL_DIR/unc_1.csv" "$AL_DIR/selectedIDS.csv"
    WORK_DIR="${MODEL_BASE}_unc_${UNC_METHOD}_1"
    RANDOM_AL_WORK_DIR="${MODEL_BASE}_1"
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
        --n-to-add "2" \
        --output-csv "$AL_DIR/uncertain_to_add.csv"\
        --uncertainty-method "$UNC_METHOD"
    PREV_IDS="$AL_DIR/unc_1.csv"
    NEXT_IDS="$AL_DIR/unc_3.csv"
    # 5. Compute uncertainty and make new splits (repeat for 3, 5, 10%)
        # Save header
    head -n 1 "$PREV_IDS" > "$NEXT_IDS"
    # Save all data, skipping headers, sort, uniq, and append
    (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
    for round in 0 1 2; do
        NEXT_PERC="${PERCENTAGES[$((round+2))]}"
        CURRENT_PERC="${PERCENTAGES[$((round+1))]}"
        ADD_PERC="${UNC_PERCENTS[$round]}"
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
            --n-to-add "$ADD_PERC" \
            --output-csv "$AL_DIR/uncertain_to_add.csv" \
            --uncertainty-method "$UNC_METHOD"

        # Make next split
        # Save header
        head -n 1 "$PREV_IDS" > "$NEXT_IDS"
        # Save all data, skipping headers, sort, uniq, and append
        (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
    done

      CHECKPOINT_DIR="${IDS_CHECKPOINT[$i]}"
    mkdir -p "$CHECKPOINT_DIR"
    cp "$AL_DIR/unc_1.csv" "$CHECKPOINT_DIR/unc_1.csv"
    cp "$AL_DIR/unc_3.csv" "$CHECKPOINT_DIR/unc_3.csv"
    cp "$AL_DIR/unc_5.csv" "$CHECKPOINT_DIR/unc_5.csv"
    cp "$AL_DIR/unc_10.csv" "$CHECKPOINT_DIR/unc_10.csv"
    echo "Copied split csvs to $CHECKPOINT_DIR"
done



MODEL_BASES=(
    "/home/aleksandar/mmdetdinov3"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py"
)

CONFIG_BASES_VAL=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3_train_val.py"
)

IDS_CHECKPOINT=(
    "/home/aleksandar/activeLearningIDS/dinov3"
)

AL_DIR="/home/aleksandar/activeLearningIDS"
DEVICE=3
PERCENTAGES=(1 3 5 10 0)
UNC_PERCENTS=(2 5 0)  # how much to add each round: 3→5 (add 2%), 5→10 (add 5%), 10 is last
UNC_METHOD="leastconf" # choose between margin or leastconf
for i in "${!MODEL_BASES[@]}"; do
    MODEL_BASE="${MODEL_BASES[$i]}"
    CONFIG_BASE="${CONFIG_BASES[$i]}"
    CONFIG_BASE_VAL="${CONFIG_BASES_VAL[$i]}"
    echo "============================"
    echo "Running for Model: $MODEL_BASE"
    echo "Using Config: $CONFIG_BASE"
    echo "============================"

    # 1. Copy unc_1.csv as starting point
    cp "$AL_DIR/random_1.csv" "$AL_DIR/unc_1.csv"
    cp "$AL_DIR/unc_1.csv" "$AL_DIR/selectedIDS.csv"
    WORK_DIR="${MODEL_BASE}_unc_${UNC_METHOD}_1"
    RANDOM_AL_WORK_DIR="${MODEL_BASE}_1"
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
        --n-to-add "2" \
        --output-csv "$AL_DIR/uncertain_to_add.csv"\
        --uncertainty-method "$UNC_METHOD"
    PREV_IDS="$AL_DIR/unc_1.csv"
    NEXT_IDS="$AL_DIR/unc_3.csv"
    # 5. Compute uncertainty and make new splits (repeat for 3, 5, 10%)
        # Save header
    head -n 1 "$PREV_IDS" > "$NEXT_IDS"
    # Save all data, skipping headers, sort, uniq, and append
    (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
    for round in 0 1 2; do
        NEXT_PERC="${PERCENTAGES[$((round+2))]}"
        CURRENT_PERC="${PERCENTAGES[$((round+1))]}"
        ADD_PERC="${UNC_PERCENTS[$round]}"
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
            --n-to-add "$ADD_PERC" \
            --output-csv "$AL_DIR/uncertain_to_add.csv" \
            --uncertainty-method "$UNC_METHOD"

        # Make next split
        # Save header
        head -n 1 "$PREV_IDS" > "$NEXT_IDS"
        # Save all data, skipping headers, sort, uniq, and append
        (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
    done

      CHECKPOINT_DIR="${IDS_CHECKPOINT[$i]}"
    mkdir -p "$CHECKPOINT_DIR"
    cp "$AL_DIR/unc_1.csv" "$CHECKPOINT_DIR/unc_1.csv"
    cp "$AL_DIR/unc_3.csv" "$CHECKPOINT_DIR/unc_3.csv"
    cp "$AL_DIR/unc_5.csv" "$CHECKPOINT_DIR/unc_5.csv"
    cp "$AL_DIR/unc_10.csv" "$CHECKPOINT_DIR/unc_10.csv"
    echo "Copied split csvs to $CHECKPOINT_DIR"
done



MODEL_BASES=(
    "/home/aleksandar/mmdetdinov3"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py"
)

CONFIG_BASES_VAL=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3_train_val.py"
)

IDS_CHECKPOINT=(
    "/home/aleksandar/activeLearningIDS/dinov3"
)

AL_DIR="/home/aleksandar/activeLearningIDS"
DEVICE=3
PERCENTAGES=(1 3 5 10 0)
UNC_PERCENTS=(2 5 0)  # how much to add each round: 3→5 (add 2%), 5→10 (add 5%), 10 is last
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
    cp "$AL_DIR/random_1.csv" "$AL_DIR/unc_1.csv"
    cp "$AL_DIR/unc_1.csv" "$AL_DIR/selectedIDS.csv"
    WORK_DIR="${MODEL_BASE}_unc_${UNC_METHOD}_1"
    RANDOM_AL_WORK_DIR="${MODEL_BASE}_1"
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
        --n-to-add "2" \
        --output-csv "$AL_DIR/uncertain_to_add.csv"\
        --uncertainty-method "$UNC_METHOD"
    PREV_IDS="$AL_DIR/unc_1.csv"
    NEXT_IDS="$AL_DIR/unc_3.csv"
    # 5. Compute uncertainty and make new splits (repeat for 3, 5, 10%)
        # Save header
    head -n 1 "$PREV_IDS" > "$NEXT_IDS"
    # Save all data, skipping headers, sort, uniq, and append
    (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
    for round in 0 1 2; do
        NEXT_PERC="${PERCENTAGES[$((round+2))]}"
        CURRENT_PERC="${PERCENTAGES[$((round+1))]}"
        ADD_PERC="${UNC_PERCENTS[$round]}"
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
            --n-to-add "$ADD_PERC" \
            --output-csv "$AL_DIR/uncertain_to_add.csv" \
            --uncertainty-method "$UNC_METHOD"

        # Make next split
        # Save header
        head -n 1 "$PREV_IDS" > "$NEXT_IDS"
        # Save all data, skipping headers, sort, uniq, and append
        (tail -n +2 "$PREV_IDS"; tail -n +2 "$AL_DIR/uncertain_to_add.csv") | sort -n | uniq >> "$NEXT_IDS"

        echo "Created $NEXT_IDS"
    done

      CHECKPOINT_DIR="${IDS_CHECKPOINT[$i]}"
    mkdir -p "$CHECKPOINT_DIR"
    cp "$AL_DIR/unc_1.csv" "$CHECKPOINT_DIR/unc_1.csv"
    cp "$AL_DIR/unc_3.csv" "$CHECKPOINT_DIR/unc_3.csv"
    cp "$AL_DIR/unc_5.csv" "$CHECKPOINT_DIR/unc_5.csv"
    cp "$AL_DIR/unc_10.csv" "$CHECKPOINT_DIR/unc_10.csv"
    echo "Copied split csvs to $CHECKPOINT_DIR"
done




MODEL_BASES=(
    "/home/aleksandar/mmdetdinov3"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py"
)

CONFIG_BASES_VAL=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3_train_val.py"
)

IDS_CHECKPOINT=(
    "/home/aleksandar/activeLearningIDS/dinov3"
)

# Active learning IDs dir
AL_DIR="/home/aleksandar/activeLearningIDS"
# GPU device
DEVICE=3

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