#!/usr/bin/env bash


# ========= USER SETTINGS =========
ANALYSIS_DIR="/home/aleksandar/analysis"
PY=python
DEVICE=1              # GPU for test/feature extraction
KNN_LIMIT=5000
KNN_K=20
TSNE_SAMPLES=8000      # we reuse the same NPZ as kNN, just visualize it
ATTN_NUM=64            # number of attention overlays
CKA_MAX_BATCHES=10     # batches to collect activations (speed/VRAM trade-off)

# ====== MODELS / CONFIGS / WORKDIRS ======
# Add/remove rows 1:1 across the 3 arrays below.
# MODEL_TAG is just a short name used in folders/logs.
MODEL_BASES=(
    "/home/aleksandar/faster_rcnn_rvsa_l_800_mae_mtp_dior"
    "/home/aleksandar/mmdetdino784LowLR"
    "/home/aleksandar/mmdetsatmae"
    "/home/aleksandar/mmdetscalemae"
    "/home/aleksandar/mmdetdinolarge784"
    "/home/aleksandar/mmdetdinov3"
)
MODEL_TAGS=(
  "MTP"
  "dinov2_vitB"
  "satmae"
  "scalemae"
  "dinov2vitL"
  "dinov3"     # <-- uncomment when you add the config + workdir
)

CONFIG_BASES=(
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge.py"
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py"
)

# ====== OPTIONAL: a “reference” model for CKA (compare everyone to this one) ======
CKA_REF_IDX=0   # compare against MODEL_TAGS[0]; change if you like

# ========= DO NOT EDIT BELOW UNLESS NEEDED =========

mkdir -p "$ANALYSIS_DIR"

# helper: pick latest epoch_*.pth from a dir
latest_ckpt() {
  local dir="$1"
  local ckpt
  ckpt=$(ls -1t "$dir"/epoch_*.pth 2>/dev/null | head -n 1 || true)
  echo "$ckpt"
}



# Run per-model: kNN feature dump (+acc), linear probe, t-SNE, attention overlays.
for i in "${!MODEL_TAGS[@]}"; do
  tag="${MODEL_TAGS[$i]}"
  cfg="${CONFIG_BASES[$i]}"
  wdir="${MODEL_BASES[$i]}"
  ckpt="$wdir/epoch_12.pth"

  if [[ -z "$ckpt" ]]; then
    echo "[SKIP] $tag (no ckpt)"
    continue
  fi

  out_root="$ANALYSIS_DIR/$tag"
  mkdir -p "$out_root"
  log="$out_root/run.log"

  echo "========== $tag ==========" | tee -a "$log"
  echo "CFG:   $cfg" | tee -a "$log"
  echo "CKPT:  $ckpt" | tee -a "$log"
  echo "OUT:   $out_root" | tee -a "$log"

  # kNN probe (+feature dump)
  CUDA_VISIBLE_DEVICES=$DEVICE $PY tools/rep_knn_probe.py \
    --config "$cfg" --ckpt "$ckpt" \
    --split train --limit $KNN_LIMIT --k $KNN_K \
    --out "$out_root/feats_train.npz" 2>&1 | tee -a "$log"

  # linear probe
  CUDA_VISIBLE_DEVICES=$DEVICE $PY tools/rep_linear_probe.py \
  --config "$cfg" --ckpt "$ckpt" \
  --limit $KNN_LIMIT --device "cuda" \
  --out "$out_root/linear_probe.txt" 2>&1 | tee -a "$log"


  # t-SNE
  # t-SNE  (now uses config+ckpt directly; no NPZ needed)
  CUDA_VISIBLE_DEVICES=$DEVICE $PY tools/rep_tsne.py \
    --config "$cfg" --ckpt "$ckpt" \
    --split train --limit $TSNE_SAMPLES \
    --out "$out_root/tsne_train.png" \
    --perplexity 30 --n-iter 1500 2>&1 | tee -a "$log"




done

# CKA: compare each model against reference
ref_tag="${MODEL_TAGS[$CKA_REF_IDX]}"
ref_cfg="${CONFIG_BASES[$CKA_REF_IDX]}"
refwdir="${MODEL_BASES[$CKA_REF_IDX]}"
ref_ckpt="$refwdir/epoch_12.pth"

if [[ -n "$ref_ckpt" ]]; then
  for i in "${!MODEL_TAGS[@]}"; do
    tag="${MODEL_TAGS[$i]}"
    cfg="${CONFIG_BASES[$i]}"
    wdir="${MODEL_BASES[$i]}"
    ckpt="$wdir/epoch_12.pth"
    if [[ -z "$ckpt" || "$i" -eq "$CKA_REF_IDX" ]]; then
      continue
    fi
    out_root="$ANALYSIS_DIR/$tag"
    log="$out_root/run.log"
    echo "CKA vs ${ref_tag}" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$DEVICE $PY tools/rep_cka.py \
      --configA "$ref_cfg" --ckptA "$ref_ckpt" \
      --configB "$cfg"     --ckptB "$ckpt" \
      --split val --limit 8000 --max-batches $CKA_MAX_BATCHES \
      --out "$out_root/cka_vs_${ref_tag}.npy" 2>&1 | tee -a "$log"
  done
else
  echo "[WARN] CKA skipped: no reference ckpt found for ${ref_tag}"
fi

echo "=== All done. Outputs in $ANALYSIS_DIR ==="