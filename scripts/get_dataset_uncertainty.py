import argparse
import pandas as pd
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uncertainty-csv', type=str, required=True,
                        help='CSV with columns id,uncertainty_margin, sorted lowest first')
    parser.add_argument('--model-path', type=str, required=True,
                        help='pth path')
    parser.add_argument('--n-to-add', type=int, required=True,
                        help='How many PERCENT to add, e.g., 2 for 2 percent of total')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Where to write output CSV of selected IDs')
    parser.add_argument('--uncertainty-method', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.model_path, 'rb') as f:
        results = pickle.load(f)

    uncertainty_scores = []

    for img_idx, res in enumerate(results):
        img_id = res.get('img_id', img_idx)  # fallback to idx if not present
        scores = res['pred_instances']['scores'].cpu().numpy()  # tensor to numpy

        if len(scores) == 0:
            # No detections, high uncertainty
            uncertainty = 1.0
        elif len(scores) == 1:
            # Only one detection, uncertainty is 1 - score
            uncertainty = 1.0 - scores[0]/1000
            if args.uncertainty_method == "count":
                uncertainty = 1/1000
        else:
            sorted_scores = np.sort(scores)[::-1]
            
            if args.uncertainty_method == "margin":
                uncertainty = sorted_scores[0] - sorted_scores[1]  # margin uncertainty
            elif args.uncertainty_method == "leastconf":
                uncertainty = 1 - sorted_scores[0]
            elif args.uncertainty_method == "count":
                uncertainty = (sorted_scores < 0.3).sum()/1000
            elif args.uncertainty_method == "average":
                uncertainty = 1 - sorted_scores.mean()
        uncertainty_scores.append((img_id, uncertainty))

    # Lower margin = more uncertain!
    uncertainty_scores.sort(key=lambda x: x[1], reverse=True)  # sort by uncertainty

    unc_df = pd.DataFrame(uncertainty_scores, columns=['id', 'uncertainty_margin'])
    unc_df = unc_df[['id']]
    unc_df.to_csv(args.uncertainty_csv, index=False)
 

    # Remove already selected
    filtered = unc_df.reset_index(drop=True)

    # Compute how many to add
    n_total = 11725
    n_add = int(n_total * args.n_to_add // 100)
    n_add = max(1, n_add)  # Always add at least one


    import os
    prev_ids_path = '/home/aleksandar/activeLearningIDS/selectedIDS.csv'
    
    if os.path.exists(prev_ids_path):
        prev_selected = pd.read_csv(prev_ids_path)
        prev_set = set(prev_selected['id'])
        # Only keep truly new IDs
        truly_new = filtered[~filtered['id'].isin(prev_set)]
        if len(truly_new) < len(filtered):
            print(f"Removed {len(filtered) - len(truly_new)} overlapping IDs.")
        filtered = truly_new.copy()
 
    selected = filtered.head(n_add)

    # Save to output, only the id column
    selected[['id']].to_csv(args.output_csv, index=False)
    print(f"Selected {len(selected)} new uncertain samples (top {args.n_to_add}% of pool).")

if __name__ == '__main__':
    main()
