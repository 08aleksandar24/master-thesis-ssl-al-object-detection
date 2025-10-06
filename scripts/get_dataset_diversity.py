# tools/get_dataset_diversity.py
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, required=True,
                  help='Pickle with list of per-image predictions (mmdet test output).')
    p.add_argument('--output-csv', type=str, required=True,
                  help='Where to write selected IDs (CSV with column id).')
    p.add_argument('--n-to-add', type=int, required=True,
                  help='How many PERCENT of TOTAL dataset to add (e.g., 2).')
    p.add_argument('--num-classes', type=int, default=20,
                  help='For class-histogram feature size.')
    p.add_argument('--distance', type=str, default='cosine',
                  choices=['cosine', 'euclidean'], help='Clustering distance.')
    p.add_argument('--prev-ids', type=str, default='/home/aleksandar/activeLearningIDS/selectedIDS.csv',
                  help='CSV with previously selected IDs to dedupe against.')
    p.add_argument('--rank-out', type=str, default=None,
                  help='Optional: write cluster assignments here (id, cluster_id).')
    return p.parse_args()


def build_image_feature(res, num_classes, low_thr=0.3):
    scores = res['pred_instances']['scores'].cpu().numpy()
    labels = res['pred_instances']['labels'].cpu().numpy()

    hist = np.zeros(num_classes, dtype=float)
    if len(labels):
        vals, cnts = np.unique(labels, return_counts=True)
        vals = vals[vals < num_classes]
        hist[vals] = cnts[:len(vals)]
        hist = hist / (hist.sum() + 1e-9)

    det_count = float(len(scores))
    if len(scores):
        s_mean = float(scores.mean())
        s_std  = float(scores.std())
        s_min  = float(scores.min())
        s_max  = float(scores.max())
        frac_low = float((scores < low_thr).sum()) / len(scores)
    else:
        s_mean = 0.0; s_std = 0.0; s_min = 1.0; s_max = 0.0; frac_low = 1.0

    return np.concatenate([hist, np.array([det_count, s_mean, s_std, s_min, s_max, frac_low], dtype=float)])


def select_by_clustering(ids, feats, n_clusters, metric='cosine'):
    X = normalize(feats) if metric == 'cosine' else feats
    n_clusters = min(n_clusters, len(ids))
    if n_clusters <= 0:
        return []

    # NOTE: requires scikit-learn >= 1.2 for `metric=...`. On older versions, use affinity=... instead.
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric=metric)
    labels = ac.fit_predict(X)

    selected = []
    for k in range(n_clusters):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue
        cluster_X = X[idxs]
        centroid = cluster_X.mean(axis=0, keepdims=True)
        d = pairwise_distances(cluster_X, centroid, metric=metric).ravel()
        medoid_local = idxs[d.argmin()]
        selected.append(ids[medoid_local])
    return selected


def main():
    args = parse_args()

    # Load predictions; defines TOTAL SIZE
    with open(args.model_path, 'rb') as f:
        results = pickle.load(f)
    total_size = len(results)
    target_add = max(1, int((total_size * args.n_to_add) // 100))

    # Build ids + features
    ids, feats = [], []
    for img_idx, res in enumerate(results):
        img_id = res.get('img_id', img_idx)
        ids.append(img_id)
        feats.append(build_image_feature(res, args.num_classes))
    feats = np.stack(feats)

    # Candidate pool: full set, or top-X% most-uncertain
    pool_ids = np.array(ids)
    pool_feats = feats


    # Dedupe against previously selected
    if os.path.exists(args.prev_ids):
        prev = pd.read_csv(args.prev_ids)
        prev_set = set(prev['id'])
        keep = np.array([i not in prev_set for i in pool_ids], dtype=bool)
        removed = int((~keep).sum())
        if removed > 0:
            print(f"Removed {removed} overlapping IDs from previous selections.")
        pool_ids = pool_ids[keep]
        pool_feats = pool_feats[keep]

    if len(pool_ids) == 0:
        print("No candidates left to select.")
        pd.DataFrame({'id': []}).to_csv(args.output_csv, index=False)
        return

    n_add = min(target_add, len(pool_ids))
    selected_ids = select_by_clustering(pool_ids, pool_feats, n_add, metric=args.distance)

    if args.rank_out is not None:
        X = normalize(pool_feats) if args.distance == 'cosine' else pool_feats
        ac = AgglomerativeClustering(n_clusters=n_add, linkage='average', metric=args.distance)
        labs = ac.fit_predict(X)
        pd.DataFrame({'id': pool_ids, 'cluster_id': labs}).to_csv(args.rank_out, index=False)

    pd.DataFrame({'id': selected_ids}).to_csv(args.output_csv, index=False)
    print(f"Selected {len(selected_ids)} new samples via clustering "
          f"({args.distance}); target {target_add} ({args.n_to_add}% of total {total_size}).")


if __name__ == '__main__':
    main()
