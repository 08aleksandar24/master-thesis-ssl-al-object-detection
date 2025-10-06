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
for i in "${!MODEL_BASES[@]}"; do
  MODEL_BASE="${MODEL_BASES[$i]}"
  CONFIG_BASE="${CONFIG_BASES[$i]}"
  WORK_DIR_20="${MODEL_BASE}_20"
  CKPT_20="${WORK_DIR_20}/epoch_12.pth"
  OUT_20="${WORK_DIR_20}/results_epoch_12.pkl"

  CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE" "$CKPT_20" --out "$OUT_20"

 
done