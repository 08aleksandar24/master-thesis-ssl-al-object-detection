import argparse, os, torch, numpy as np
from mmengine.config import Config
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmdet.registry import MODELS, DATASETS
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
        # ---- unify batch into (B,C,H,W) tensor and a list of data_samples ----
        if isinstance(batch, list):
            # pseudo_collate returns list[dict]
            imgs = torch.stack([
                (s['inputs'][0] if isinstance(s['inputs'], (list, tuple)) else s['inputs'])
                for s in batch
            ], dim=0)
            data_samples = [s.get('data_samples', None) for s in batch]
        else:
            # dict with keys 'inputs', 'data_samples'
            x = batch['inputs']
            if isinstance(x, (list, tuple)):
                imgs = torch.stack(x, dim=0)
            else:
                imgs = x.unsqueeze(0)
            ds = batch.get('data_samples', None)
            data_samples = list(ds) if isinstance(ds, (list, tuple)) else [ds]

        # ---- preprocess (DetDataPreprocessor equivalent) ----
        imgs = imgs.to(device, non_blocking=True).float()
        if bgr_to_rgb:
            imgs = imgs[:, [2, 1, 0], ...]
        imgs = (imgs - mean) / std

        # ---- forward backbone ----
        outs = backbone(imgs)
        if isinstance(outs, (list, tuple)):
            x = outs[-1]
            x = x.mean(dim=(2, 3)) if x.dim() == 4 else x.mean(dim=1)
        else:
            x = outs
        feats.append(x.detach().cpu())

        # ---- image-level label proxy ----
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
    ap.add_argument('--limit', type=int, default=5000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)

    # --- dataset splits ---
    dl_tr_cfg = cfg.train_dataloader
    dl_va_cfg = cfg.val_dataloader

    ds_tr = DATASETS.build(dl_tr_cfg['dataset'])
    ds_va = DATASETS.build(dl_va_cfg['dataset'])
    if args.limit:
        ds_tr = torch.utils.data.Subset(ds_tr, range(min(args.limit, len(ds_tr))))
        ds_va = torch.utils.data.Subset(ds_va, range(min(args.limit, len(ds_va))))

    loader_tr = DataLoader(
        ds_tr,
        batch_size=dl_tr_cfg.get('batch_size', 2),
        num_workers=dl_tr_cfg.get('num_workers', 4),
        collate_fn=pseudo_collate,
        shuffle=False,
        pin_memory=True,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=dl_va_cfg.get('batch_size', 2),
        num_workers=dl_va_cfg.get('num_workers', 4),
        collate_fn=pseudo_collate,
        shuffle=False,
        pin_memory=True,
    )

    # --- build & load model ---
    model = MODELS.build(cfg.model)
    _ = load_checkpoint(model, args.ckpt, map_location='cpu')
    model.to(args.device).eval()

    prep = cfg.model['data_preprocessor']
    mean = prep.get('mean', [123.675, 116.28, 103.53])
    std  = prep.get('std',  [58.395, 57.12, 57.375])
    bgr_to_rgb = prep.get('bgr_to_rgb', True)

    backbone = get_backbone(model).to(args.device)

    # --- extract features ---
    print("Extracting train features...")
    Xtr_t, ytr = extract_feats(backbone, loader_tr, args.device, mean, std, bgr_to_rgb)
    print("Extracting val features...")
    Xva_t, yva = extract_feats(backbone, loader_va, args.device, mean, std, bgr_to_rgb)

    Xtr = normalize(Xtr_t, dim=1).cpu().numpy()
    Xva = normalize(Xva_t, dim=1).cpu().numpy()
    ytr, yva = np.array(ytr), np.array(yva)

    # save feats
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)

    # --- linear probe ---
    mask_tr = ytr >= 0
    mask_va = yva >= 0
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, verbose=1)
    clf.fit(Xtr[mask_tr], ytr[mask_tr])
    preds = clf.predict(Xva[mask_va])
    acc = accuracy_score(yva[mask_va], preds)
    print(f"Linear probe accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()
