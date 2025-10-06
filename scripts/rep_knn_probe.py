import argparse, os, torch, numpy as np
from mmengine.config import Config
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmdet.registry import MODELS, DATASETS
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

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
        # handle both list-of-samples and dict batches
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

        # --- preprocess (what DetDataPreprocessor would do) ---
        # imgs is uint8 CHW; convert to float32
        imgs = imgs.to(device, non_blocking=True).float()
        # channel order if needed
        if bgr_to_rgb:
            imgs = imgs[:, [2, 1, 0], ...]
        # normalize with 0-255 means/stds (do NOT divide by 255 here)
        imgs = (imgs - mean) / std
        # ------------------------------------------------------

        # forward backbone
        outs = backbone(imgs)
        if isinstance(outs, (list, tuple)):
            x = outs[-1]
            x = x.mean(dim=(2, 3)) if x.dim() == 4 else x.mean(dim=1)
        else:
            x = outs
        feats.append(x.detach().cpu())

        # quick image-level label proxy (if GT present)
        for ds in data_samples:
            if ds is None or not hasattr(ds, 'gt_instances') or ds.gt_instances is None or len(ds.gt_instances) == 0:
                labels.append(-1)
            else:
                lab = ds.gt_instances.labels.cpu().numpy()
                labels.append(int(np.bincount(lab).argmax()))
    return torch.cat(feats, dim=0), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', choices=['train','val'], default='train')
    ap.add_argument('--limit', type=int, default=5000)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)

    # pick a split that actually has GT if you want kNN labels
    dl_cfg = cfg.train_dataloader if args.split == 'train' else cfg.val_dataloader
    ds = DATASETS.build(dl_cfg['dataset'])
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))

    loader = DataLoader(
        ds,
        batch_size=dl_cfg.get('batch_size', 2),
        num_workers=dl_cfg.get('num_workers', 4),
        collate_fn=pseudo_collate,   # returns list[dict]
        shuffle=False,
        pin_memory=True,
    )

    # build & load model
    model = MODELS.build(cfg.model)
    _ = load_checkpoint(model, args.ckpt, map_location='cpu')
    model.to(args.device).eval()
    prep = cfg.model['data_preprocessor']
    mean = prep.get('mean', [123.675, 116.28, 103.53])
    std  = prep.get('std',  [58.395, 57.12, 57.375])
    bgr_to_rgb = prep.get('bgr_to_rgb', True)

    backbone = get_backbone(model).to(args.device)
    feats, labels = extract_feats(backbone, loader, args.device, mean, std, bgr_to_rgb)

    feats = normalize(feats, dim=1).cpu().numpy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, feats=feats, labels=labels)

    # quick kNN probe (only if labels exist in this split)
    from sklearn.neighbors import KNeighborsClassifier
    mask = labels >= 0
    if mask.sum() > 0:
        knn = KNeighborsClassifier(n_neighbors=args.k, metric='cosine')
        knn.fit(feats[mask], labels[mask])
        acc = (knn.predict(feats[mask]) == labels[mask]).mean()
        print(f'kNN@{args.k} accuracy: {acc:.4f}')
    else:
        print('No GT labels in this split (maybe test_mode=True). Skipping kNN.')

if __name__ == '__main__':
    main()
