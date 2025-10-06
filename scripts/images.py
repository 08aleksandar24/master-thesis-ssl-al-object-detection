# make_panels.py
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:\Mag_Modeli\analysisImages"   # <-- change if needed
SUBFOLDERS = {
    "Dinov3": "Dinov3",
    "MTP": "MTP",
    "ScaleMAE": "ScaleMAE",
}
OUTPUT_DIR = Path(ROOT) / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Try a nice font; fallback to default
def load_font(size=26):
    for fp in ["arial.ttf", "seguiemj.ttf", "segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT = load_font(26)

def index_by_stem(folder: Path):
    mapping = {}
    for p in folder.iterdir():
        if p.suffix.lower() in ALLOWED_EXTS and p.is_file():
            mapping.setdefault(p.stem, p)
    return mapping

def draw_label_bar(tile: Image.Image, text: str, bar_h=34, pad=6):
    """Draw a translucent bar and centered label at the top of the tile."""
    if tile.mode != "RGBA":
        base = tile.convert("RGBA")
    else:
        base = tile.copy()
    W, H = base.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # bar
    odraw.rectangle([0, 0, W, bar_h], fill=(0, 0, 0, 140))

    # text centered
    t_w, t_h = odraw.textbbox((0, 0), text, font=FONT)[2:]
    x = (W - t_w) // 2
    y = (bar_h - t_h) // 2
    # slight outline for readability
    odraw.text((x, y), text, font=FONT, fill=(255, 255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0, 255))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

def crop_halves(img: Image.Image):
    """Returns (left, right) halves."""
    W, H = img.size
    mid = W // 2
    left = img.crop((0, 0, mid, H))
    right = img.crop((mid, 0, W, H))
    return left, right

def main():
    dinov3_map = index_by_stem(Path(ROOT) / SUBFOLDERS["Dinov3"])
    mtp_map    = index_by_stem(Path(ROOT) / SUBFOLDERS["MTP"])
    scale_map  = index_by_stem(Path(ROOT) / SUBFOLDERS["ScaleMAE"])

    common = set(dinov3_map) & set(mtp_map) & set(scale_map)
    if not common:
        print("No common filenames across Dinov3, MTP, ScaleMAE. Nothing to do.")
        return

    processed, skipped = 0, 0
    for stem in sorted(common):
        try:
            im_mtp = Image.open(mtp_map[stem]).convert("RGB")
            im_dv3 = Image.open(dinov3_map[stem]).convert("RGB")
            im_scl = Image.open(scale_map[stem]).convert("RGB")

            # Expect 1600x800; crop halves
            gt_tile, mtp_pred = crop_halves(im_mtp)   # left is GT, right is MTP pred
            _, dv3_pred = crop_halves(im_dv3)         # right is DINOv3 pred
            _, scl_pred = crop_halves(im_scl)         # right is ScaleMAE pred

            # Ensure each tile is 800x800 (resize if needed)
            target = (800, 800)
            for name, tile in [("gt", gt_tile), ("mtp", mtp_pred), ("dv3", dv3_pred), ("scl", scl_pred)]:
                if tile.size != target:
                    tile = tile.resize(target, Image.BICUBIC)
                if name == "gt": gt_tile = tile
                elif name == "mtp": mtp_pred = tile
                elif name == "dv3": dv3_pred = tile
                else: scl_pred = tile

            # Label each tile
            gt_tile  = draw_label_bar(gt_tile,  "Ground Truth")
            mtp_pred = draw_label_bar(mtp_pred, "MTP")
            dv3_pred = draw_label_bar(dv3_pred, "DINOv3")
            scl_pred = draw_label_bar(scl_pred, "ScaleMAE")

            # Compose 1600x1600
            canvas = Image.new("RGB", (1600, 1600), (255, 255, 255))
            canvas.paste(gt_tile,  (0,   0))
            canvas.paste(mtp_pred, (800, 0))
            canvas.paste(dv3_pred, (0,   800))
            canvas.paste(scl_pred, (800, 800))

            out_path = OUTPUT_DIR / f"{stem}_panel.png"
            canvas.save(out_path, "PNG", optimize=True)
            processed += 1
        except Exception as e:
            skipped += 1
            print(f"Skip {stem}: {e}")

    print(f"Done. Saved {processed} panels to {OUTPUT_DIR}. Skipped {skipped}.")

if __name__ == "__main__":
    main()
