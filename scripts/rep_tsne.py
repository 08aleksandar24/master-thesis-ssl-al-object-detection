# tools/rep_tsne.py
import argparse, os, torch, numpy as np
from mmengine.config import Config
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmdet.registry import MODELS, DATASETS
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_backbone(model):
    if hasattr(model, 'backbone'):
        return model.backbone
    raise AttributeError('No backbone found on the model.')

@torch.no_grad()
def extract_feats(backbone, loader, device, mean, std, bgr_to_rgb):
    backbone.eval()
    feats, labels = [], []

    mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, 3, 1, 1)

    for batch in loader:
        # pseudo_collate -> list[dict] with keys 'inputs', 'data_samples'
        if isinstance(batch, list):
            imgs = torch.stack([
                (s['inputs'][0] if isinstance(s['inputs'], (list, tuple)) else s['inputs'])
                for s in batch
            ], dim=0)
            data_samples = [s.get('data_samples', None) for s in batch]
        else:
            x = batch['inputs']
            if isinstance(x, (list, tuple)):
                imgs = torch.stack(x, dim=0)
            else:
                imgs = x.unsqueeze(0)
            ds = batch.get('data_samples', None)
            data_samples = list(ds) if isinstance(ds, (list, tuple)) else [ds]

        # Preprocess (DetDataPreprocessor equivalent)
        imgs = imgs.to(device, non_blocking=True).float()
        if bgr_to_rgb:
            imgs = imgs[:, [2, 1, 0], ...]
        imgs = (imgs - mean) / std

        outs = backbone(imgs)
        if isinstance(outs, (list, tuple)):
            x = outs[-1]
            x = x.mean(dim=(2, 3)) if x.dim() == 4 else x.mean(dim=1)
        else:
            x = outs
        feats.append(x.detach().cpu())

        # quick per-image label proxy from GT (if available)
        for ds in data_samples:
            if ds is None or not hasattr(ds, 'gt_instances') or ds.gt_instances is None or len(ds.gt_instances) == 0:
                labels.append(-1)
            else:
                lab = ds.gt_instances.labels.cpu().numpy()
                labels.append(int(np.bincount(lab).argmax()))

    return torch.cat(feats, dim=0).numpy(), np.array(labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='MMDet config path')
    ap.add_argument('--ckpt', required=True, help='Checkpoint path')
    ap.add_argument('--split', choices=['train','val'], default='val')
    ap.add_argument('--limit', type=int, default=3000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True, help='Output PNG path')
    ap.add_argument('--perplexity', type=float, default=30.0)
    ap.add_argument('--n-iter', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)

    # choose split
    dl_cfg = cfg.train_dataloader if args.split == 'train' else cfg.val_dataloader
    dataset = DATASETS.build(dl_cfg['dataset'])
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=dl_cfg.get('batch_size', 2),
        num_workers=dl_cfg.get('num_workers', 4),
        collate_fn=pseudo_collate,
        shuffle=False,
        pin_memory=True,
    )

    # build & load model
    model = MODELS.build(cfg.model)
    _ = load_checkpoint(model, args.ckpt, map_location='cpu')
    model.to(args.device).eval()

    prep = cfg.model.get('data_preprocessor', {})
    mean = prep.get('mean', [123.675, 116.28, 103.53])
    std  = prep.get('std',  [58.395, 57.12, 57.375])
    bgr_to_rgb = prep.get('bgr_to_rgb', True)

    backbone = get_backbone(model).to(args.device)

    # extract features
    print(f'Extracting features from {args.split} split (limit={args.limit})...')
    X, y = extract_feats(backbone, loader, args.device, mean, std, bgr_to_rgb)

    # keep only images with labels if possible; else just color all same
    mask = y >= 0
    if mask.sum() == 0:
        print('No GT labels found in this split. t-SNE will plot all points with one color.')
        y_plot = np.zeros(len(y), dtype=int)
        X_plot = X
    else:
        y_plot = y[mask]
        X_plot = X[mask]

    # t-SNE
    print('Running t-SNE...')
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        init='pca',
        learning_rate='auto',
        metric='cosine',
        random_state=args.seed,
        verbose=1,
    )
    X2 = tsne.fit_transform(X_plot)

    # plot
    plt.figure(figsize=(8, 7), dpi=150)
    if y_plot.ndim == 1:
        scatter = plt.scatter(X2[:,0], X2[:,1], c=y_plot, s=6, alpha=0.7, cmap='tab20')
        # optional legend (can be cluttered with many classes)
        # plt.legend(*scatter.legend_elements(), loc='best', fontsize='small', title='Class')
    else:
        plt.scatter(X2[:,0], X2[:,1], s=6, alpha=0.7, c='gray')
    plt.title(f't-SNE ({args.split}) - {os.path.basename(args.config)}')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'Saved t-SNE to {args.out}')

if __name__ == '__main__':
    main()
