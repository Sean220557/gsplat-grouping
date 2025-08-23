from pathlib import Path
import argparse
import numpy as np

from PIL import Image
import imageio.v2 as imageio
import cv2

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def load_image_rgb(p):
    img = imageio.imread(p)
    if img.ndim == 2:
        img = np.stack([img, img, img], -1)
    if img.shape[2] == 4:
        img = img[..., :3]
    return img

def draw_overlay(rgb, label, alpha=0.5):
    rng = np.random.RandomState(0)
    K = int(label.max())
    cmap = rng.randint(0, 255, (max(K + 1, 256), 3), dtype=np.uint8)
    cmap[0] = np.array([30, 30, 30], dtype=np.uint8)
    color = cmap[label]
    out = (alpha * color + (1 - alpha) * rgb).astype(np.uint8)
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def generate_label_from_sam(sam, img_rgb, strict, topk, min_area, morph_close, morph_ksize):
    if strict:
        gen = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_overlap_ratio=0.341,
            box_nms_thresh=0.7,
            min_mask_region_area=min_area,
        )
    else:
        gen = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            crop_n_layers=0,
            crop_overlap_ratio=0.0,
            box_nms_thresh=0.8,
            min_mask_region_area=max(min_area // 2, 16),
        )

    anns = gen.generate(img_rgb)
    H, W = img_rgb.shape[:2]
    label = np.zeros((H, W), dtype=np.uint8)

    if len(anns) == 0:
        return label, 0

    anns.sort(key=lambda x: x.get("area", 0), reverse=True)
    kept = anns[:topk]

    cur_id = 1
    for a in kept:
        m = a["segmentation"].astype(np.uint8)
        if morph_close:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        label[(label == 0) & (m > 0)] = cur_id
        cur_id += 1
        if cur_id >= 255:
            break

    return label, (cur_id - 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="e.g. data/360_v2/garden")
    ap.add_argument("--images_dir", type=str, default=None, help="if None, use data_dir/images or images_4")
    ap.add_argument("--mask_dir", type=str, required=True, help="output dir for labelmaps")
    ap.add_argument("--sam_model", type=str, default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--sam_ckpt", type=str, required=True)
    ap.add_argument("--topk", type=int, default=12, help="keep top-K instances per image (<= 254)")
    ap.add_argument("--min_area", type=int, default=64, help="filter tiny regions before composing label")
    ap.add_argument("--morph_close", action="store_true", help="apply morphological close on masks")
    ap.add_argument("--morph_ksize", type=int, default=5)
    ap.add_argument("--exts", type=str, default="jpg,jpeg,png,JPG,PNG")
    ap.add_argument("--viz", action="store_true", help="save overlay visualization *_viz.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam] loading {args.sam_model} from {args.sam_ckpt} on {device}")
    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to(device=device)
    sam.eval()

    data_dir = Path(args.data_dir)
    if args.images_dir is None:
        if (data_dir/"images_4").exists():
            images_dir = data_dir/"images_4"
        else:
            images_dir = data_dir/"images"
    else:
        images_dir = Path(args.images_dir)

    out_dir = Path(args.mask_dir)
    ensure_dir(out_dir)

    exts = tuple(args.exts.split(","))
    files = sorted([p for p in images_dir.rglob("*") if p.suffix.replace(".","") in exts])
    if len(files) == 0:
        for e in exts:
            files += list((images_dir).glob(f"*.{e}"))
        files = sorted(files)

    print(f"[data] images: {len(files)} from {images_dir}")
    print(f"[save] masks -> {out_dir}  (topk={args.topk}, min_area={args.min_area})")

    for i, fp in enumerate(files):
        img = load_image_rgb(fp.as_posix())
        H, W = img.shape[:2]

        label, n1 = generate_label_from_sam(
            sam, img, strict=True, topk=args.topk, min_area=args.min_area,
            morph_close=args.morph_close, morph_ksize=args.morph_ksize
        )
        if n1 == 0:
            label, n2 = generate_label_from_sam(
                sam, img, strict=False, topk=args.topk, min_area=args.min_area,
                morph_close=args.morph_close, morph_ksize=args.morph_ksize
            )
            n1 = n2
        stem = fp.stem
        out_path = out_dir / f"{stem}.png"
        Image.fromarray(label, mode="L").save(out_path)

        if args.viz:
            ov = draw_overlay(img, label, alpha=0.45)
            imageio.imwrite((out_dir / f"{stem}_viz.png").as_posix(), ov)

        if (i % 10) == 0:
            fg_ratio = float((label > 0).sum()) / (H * W)
            print(f"[{i+1}/{len(files)}] {fp.name}: instances={int(label.max())}, fg_ratio={fg_ratio:.3f}")

    print("[done] SAM precompute completed.")

if __name__ == "__main__":
    main()
