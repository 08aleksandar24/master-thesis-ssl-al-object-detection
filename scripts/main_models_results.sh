#!/bin/bash

MODEL_BASES=(
    "/home/aleksandar/faster_rcnn_rvsa_l_800_mae_mtp_dior"
    "/home/aleksandar/mmdetdino784LowLR"
    "/home/aleksandar/mmdetsatmae"
    "/home/aleksandar/mmdetscalemae"
    "/home/aleksandar/mmdetdinolarge784"
    "/home/aleksandar/mmdetdinov3"
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py"
)


# GPU device
DEVICE=0




############################################
# Main loop
############################################
for i in "${!MODEL_BASES[@]}"; do
  MODEL_BASE="${MODEL_BASES[$i]}"
  CONFIG_BASE="${CONFIG_BASES[$i]}"

  echo "==========================================="
  echo "Model:     $MODEL_BASE"
  echo "Config:    $CONFIG_BASE"
  echo "==========================================="

  
  CKPT_NEXT="$MODEL_BASE/epoch_12.pth"
  TEST_OUT_NEXT="$MODEL_BASE/results_epoch_12.pkl"
  CUDA_VISIBLE_DEVICES=$DEVICE torchrun tools/test.py "$CONFIG_BASE" "$CKPT_NEXT" --out "$TEST_OUT_NEXT"

done