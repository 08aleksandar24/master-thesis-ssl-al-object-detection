import os
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm

# Paths from your setup
MODEL_BASES = [
    "/home/aleksandar/faster_rcnn_rvsa_l_800_mae_mtp_dior",
    "/home/aleksandar/mmdetsatmae",
    "/home/aleksandar/mmdetscalemae",
    "/home/aleksandar/mmdetdinolarge784",
    "/home/aleksandar/mmdetdinov3",
    "/home/aleksandar/mmdetdino784LowLR"
]

CONFIG_BASES = [
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py",
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/satmae.py",
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/scalemae.py",
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtplarge.py",
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py",
    "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinomtp.py"
]

# Location of DIOR test images
TEST_DIR = "/storage/datasets/dior/JPEGImages-test"
# Where to save visualizations
OUTPUT_ROOT = "/home/aleksandar/qualitative_results"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Score threshold for visualization
SCORE_THR = 0.7

# Loop through models
for config, model_dir in zip(CONFIG_BASES, MODEL_BASES):
    checkpoint = os.path.join(model_dir, "epoch_12.pth")  # adjust if your ckpt filename differs
    backbone_name = os.path.basename(model_dir)
    save_dir = os.path.join(OUTPUT_ROOT, backbone_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nRunning inference for {backbone_name}")
    model = init_detector(config, checkpoint)

    # Loop through test images
    for img_name in tqdm(os.listdir(TEST_DIR)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(TEST_DIR, img_name)
        out_path = os.path.join(save_dir, img_name)

        result = inference_detector(model, img_path)
        model.show_result(
            img_path,
            result,
            score_thr=SCORE_THR,
            out_file=out_path
        )

print("✅ Done. All visualizations saved to", OUTPUT_ROOT)
