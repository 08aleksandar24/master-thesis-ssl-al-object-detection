import os
import glob
import pickle
import torch
import numpy as np
import json
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------- Config ---------------- #
SEED = 42
np.random.seed(SEED)
BOOT_ITERS = 100  # Change if needed

# ---------------- Helper Functions ---------------- #
def compute_coco_ap(rec, prec, rec_thrs):
    if len(rec) == 0:
        return 0.0
    mpre = np.maximum.accumulate(prec[::-1])[::-1]
    ap = 0.0
    for t in rec_thrs:
        inds = np.searchsorted(rec, t, side='left')
        if inds < len(mpre):
            ap += mpre[inds]
        else:
            ap += 0.0
    return ap / len(rec_thrs)

def prepare_eval_data(coco_eval):
    evalImgs = coco_eval.evalImgs
    params = coco_eval.params
    iou_thrs = params.iouThrs
    max_dets = params.maxDets[-1]
    rec_thrs = params.recThrs
    cat_ids = params.catIds

    per_image_cat_data = {}
    gt_count_per_img_cat = {}

    for e in evalImgs:
        if e is None:
            continue
        img_id = e['image_id']
        cat_id = e['category_id']
        key = (img_id, cat_id)
        if key not in per_image_cat_data:
            per_image_cat_data[key] = {'scores': [], 'tp': [], 'fp': []}

        dt_scores = np.array(e['dtScores'], dtype=np.float32)
        dt_ignore = np.array(e['dtIgnore'], dtype=bool)
        dt_matches = np.array(e['dtMatches'], dtype=np.int64)

        if len(dt_scores) > max_dets:
            dt_scores = dt_scores[:max_dets]
            dt_ignore = dt_ignore[:max_dets]
            dt_matches = dt_matches[:, :max_dets]

        tp = (dt_matches > 0) & (~dt_ignore)
        fp = (dt_matches == 0) & (~dt_ignore)

        per_image_cat_data[key]['scores'].append(dt_scores)
        per_image_cat_data[key]['tp'].append(tp)
        per_image_cat_data[key]['fp'].append(fp)

    for key, data in per_image_cat_data.items():
        data['scores'] = np.concatenate(data['scores'])
        data['tp'] = np.concatenate(data['tp'], axis=1)
        data['fp'] = np.concatenate(data['fp'], axis=1)

    for e in evalImgs:
        if e is None:
            continue
        img_id = e['image_id']
        cat_id = e['category_id']
        key = (img_id, cat_id)
        if key not in gt_count_per_img_cat:
            gt_count_per_img_cat[key] = 0
        gt_count_per_img_cat[key] += np.sum(~np.array(e['gtIgnore'], dtype=bool))

    return per_image_cat_data, gt_count_per_img_cat, cat_ids, iou_thrs, rec_thrs

def compute_map_for_subset(sampled_img_ids, per_image_cat_data, gt_count_per_img_cat, cat_ids, iou_thrs, rec_thrs):
    aps_all_cats = []
    aps50_all_cats = []

    for cat_id in cat_ids:
        all_scores, all_tp, all_fp = [], [], []
        total_gt = sum(gt_count_per_img_cat.get((img_id, cat_id), 0) for img_id in sampled_img_ids)

        if total_gt == 0:
            continue

        for img_id in sampled_img_ids:
            key = (img_id, cat_id)
            if key in per_image_cat_data:
                d = per_image_cat_data[key]
                all_scores.append(d['scores'])
                all_tp.append(d['tp'])
                all_fp.append(d['fp'])

        if not all_scores:
            aps_all_cats.append(0.0)
            aps50_all_cats.append(0.0)
            continue

        all_scores = np.concatenate(all_scores)
        all_tp = np.concatenate(all_tp, axis=1)
        all_fp = np.concatenate(all_fp, axis=1)

        idx = np.argsort(-all_scores)
        all_tp = all_tp[:, idx]
        all_fp = all_fp[:, idx]

        tp_cum = np.cumsum(all_tp, axis=1)
        fp_cum = np.cumsum(all_fp, axis=1)

        rec = tp_cum / (total_gt + 1e-6)
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)

        aps = [compute_coco_ap(rec[i], prec[i], rec_thrs) for i in range(len(iou_thrs))]
        aps_all_cats.append(np.mean(aps))
        aps50_all_cats.append(aps[np.where(np.isclose(iou_thrs, 0.5))[0][0]])

    if not aps_all_cats:
        return np.nan, np.nan

    return np.mean(aps_all_cats), np.mean(aps50_all_cats)

# ---------------- Main Function ---------------- #
def process_model_dir(MODEL_DIR, ann_file):
    print(f"\nProcessing folder: {MODEL_DIR}")
    summary = {}

    for pkl_path in sorted(glob.glob(os.path.join(MODEL_DIR, "results_epoch_12.pkl"))):
        epoch_num = int(os.path.basename(pkl_path).split('_')[2].split('.')[0])
        print(f"  Epoch {epoch_num}")
        results = pickle.load(open(pkl_path, 'rb'))
        json_out = pkl_path.replace('.pkl', '_coco.json')

        dets = []
        for entry in results:
            img_id = entry['img_id']
            pred = entry['pred_instances']
            bboxes = pred['bboxes'].cpu().numpy() if torch.is_tensor(pred['bboxes']) else pred['bboxes']
            scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
            labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox.tolist()
                dets.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(score)
                })

        with open(json_out, 'w') as f:
            json.dump(dets, f)

        coco_gt = COCO(ann_file)
        if 'info' not in coco_gt.dataset:
            coco_gt.dataset['info'] = {}
        coco_dt = coco_gt.loadRes(json_out)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_official = coco_eval.stats[0]
        map50_official = coco_eval.stats[1]
        map_small = coco_eval.stats[3]
        map_medium = coco_eval.stats[4]
        map_large = coco_eval.stats[5]

        per_image_cat_data, gt_count_per_img_cat, cat_ids, iou_thrs, rec_thrs = prepare_eval_data(coco_eval)
        img_ids = coco_gt.getImgIds()

        print(f"[Validation] Vectorized vs Official:")
        val_mAP, val_mAP50 = compute_map_for_subset(img_ids, per_image_cat_data, gt_count_per_img_cat, cat_ids, iou_thrs, rec_thrs)
        print(f"  Vectorized: mAP={val_mAP:.4f}, mAP50={val_mAP50:.4f}")
        print(f"  Official:   mAP={map_official:.4f}, mAP50={map50_official:.4f}")

        print(f"Running {BOOT_ITERS} bootstrap iterations...")
        seeds = np.random.SeedSequence(SEED).spawn(BOOT_ITERS)
        seed_list = [int(s.generate_state(1)[0]) for s in seeds]

        def bootstrap_once(seed):
            np.random.seed(seed)
            sampled_img_ids = np.random.choice(img_ids, size=len(img_ids), replace=True)
            return compute_map_for_subset(sampled_img_ids, per_image_cat_data, gt_count_per_img_cat, cat_ids, iou_thrs, rec_thrs)

        bootstrap_results = Parallel(n_jobs=-1)(
            delayed(bootstrap_once)(seed_list[i]) for i in tqdm(range(BOOT_ITERS))
        )

        map_list, map50_list = zip(*bootstrap_results)
        map_arr = np.array(map_list)
        map50_arr = np.array(map50_list)

        summary[epoch_num] = {
            'official_mAP': float(map_official),
            'official_mAP50': float(map50_official),
            'official_mAP_small': float(map_small),
            'official_mAP_medium': float(map_medium),
            'official_mAP_large': float(map_large),
            'mAP_mean': float(map_arr.mean()),
            'mAP_std': float(map_arr.std()),
            'mAP_CI_5': float(np.percentile(map_arr, 5)),
            'mAP_CI_95': float(np.percentile(map_arr, 95)),
            'mAP50_mean': float(map50_arr.mean()),
            'mAP50_std': float(map50_arr.std()),
            'mAP50_CI_5': float(np.percentile(map50_arr, 5)),
            'mAP50_CI_95': float(np.percentile(map50_arr, 95)),
            'map_bootstrap': map_arr.tolist(),
            'map50_bootstrap': map50_arr.tolist()
        }

        print(summary[epoch_num])

    out_path = os.path.join(MODEL_DIR, 'bootstrap_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_path}")

# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <ann_file.json> <folder1> [<folder2> ...]")
        sys.exit(1)

    ann_file = sys.argv[1]
    model_dirs = []
    for path in sys.argv[2:]:
        if os.path.isdir(path):
            subdirs = [
                os.path.join(path, d)
                for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and os.path.isfile(os.path.join(path, d, "epoch_12.pth")) and not os.path.isfile(os.path.join(path, d, "bootstrap_summary.json"))
            ]
            model_dirs.extend(subdirs)
        else:
            print(f"Warning: {path} is not a directory. Skipping.")
    print(len(model_dirs))
    for model_dir in model_dirs:
        process_model_dir(model_dir, ann_file)
