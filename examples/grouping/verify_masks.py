import argparse, csv, os
from pathlib import Path
import glob
import numpy as np
import torch
import imageio.v2 as imageio

import sys
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES_DIR))
from datasets.colmap import Dataset as ColmapDataset, Parser as ColmapParser


def _load_images_list(images_dir: Path):
    files = sorted(glob.glob(str(images_dir / "*.jpg"))) + sorted(
        glob.glob(str(images_dir / "*.png"))
    )
    stems = [Path(f).stem for f in files]
    return stems, files


def _infer_hw_from_item(item):
    if "height" in item and "width" in item:
        return int(item["height"]), int(item["width"])
    img = item["image"]
    if isinstance(img, torch.Tensor):
        if img.dim() == 3 and img.shape[-1] in (3, 4):
            return int(img.shape[0]), int(img.shape[1])  # HWC
        if img.dim() == 3 and img.shape[0] in (1, 3, 4):
            return int(img.shape[1]), int(img.shape[2])  # CHW
    elif isinstance(img, np.ndarray):
        return int(img.shape[0]), int(img.shape[1])
    raise ValueError("Cannot infer H/W")


def _nearest_resize_mask(mask_np, H, W):
    m = torch.from_numpy(mask_np)[None, None].float()
    m = (
        torch.nn.functional.interpolate(m, size=(H, W), mode="nearest")
        .squeeze()
        .long()
        .cpu()
        .numpy()
    )
    return m


def _colorize(mask):
    m = mask.astype(np.int32)
    n = int(m.max()) + 1
    rng = np.random.RandomState(0)
    cmap = rng.randint(0, 255, size=(max(n, 256), 3), dtype=np.uint8)
    cmap[0] = (0, 0, 0)
    return cmap[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument(
        "--images_dir", default=None, help="明确指定 images_*，否则按 factor 推测"
    )
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--export_csv", default="mask_report.csv")
    ap.add_argument(
        "--export_overlays", default=None, help="导出叠加可视化目录（可选）"
    )
    ap.add_argument("--overlay_limit", type=int, default=24)
    args = ap.parse_args()

    root = Path(args.data_dir)
    masks_dir = Path(args.mask_dir)
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        cand = list(root.glob(f"images_{args.factor}"))
        if not cand:
            alt = sorted(
                [p for p in root.glob("images_*") if p.is_dir()], key=lambda p: p.name
            )
            images_dir = alt[-1] if alt else root / "images"
        else:
            images_dir = cand[0]

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"mask_dir not found: {masks_dir}")

    parser = ColmapParser(data_dir=args.data_dir, factor=args.factor)
    base = ColmapDataset(parser, split="train")

    image_stems, image_files = _load_images_list(images_dir)
    if len(image_stems) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    total = min(len(base), len(image_stems))
    print(f"Total frames (min(dataset,images_dir)): {total}")

    rows = []
    ok = zero = miss = shape_mismatch = 0

    if args.export_overlays:
        Path(args.export_overlays).mkdir(parents=True, exist_ok=True)

    for i in range(total):
        item = base[i]
        H, W = _infer_hw_from_item(item)
        stem = image_stems[i]
        mp = masks_dir / f"{stem}.png"
        status = "OK"
        nz_ratio = 0.0
        uniq = 0
        sh_mis = False

        if not mp.exists():
            status = "MISSING"
            miss += 1
        else:
            m = imageio.imread(mp.as_posix())
            if m.ndim == 3:  # 容错：若误存成 RGB
                m = m[..., 0]
            if m.shape[:2] != (H, W):
                m = _nearest_resize_mask(m, H, W)
                sh_mis = True
                shape_mismatch += 1
            nz = int((m != 0).sum())
            nz_ratio = nz / (H * W)
            uniq = int(np.unique(m).size)
            if nz == 0:
                status = "ALL_ZERO"
                zero += 1
            else:
                ok += 1

            # 叠加可视化
            if args.export_overlays and i < args.overlay_limit:
                import cv2

                # 读原图
                img_path = image_files[i]
                img = imageio.imread(img_path)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                col = _colorize(m)
                blend = (0.7 * img + 0.3 * col).astype(np.uint8)
                imageio.imwrite(
                    str(Path(args.export_overlays) / f"{i:04d}_{stem}.png"), blend
                )

        rows.append(
            {
                "index": i,
                "stem": stem,
                "status": status,
                "nonzero_ratio": f"{nz_ratio:.6f}",
                "unique_labels": uniq,
                "shape_mismatch": int(sh_mis),
                "H": H,
                "W": W,
                "mask_path": str(mp if mp.exists() else ""),
            }
        )

    print(f"OK(has FG): {ok}")
    print(f"ALL_ZERO : {zero}")
    print(f"MISSING  : {miss}")
    print(f"shape_mismatch resized: {shape_mismatch}")

    with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved: {args.export_csv}")

    if args.export_overlays:
        print(f"Overlays saved to: {args.export_overlays}")


if __name__ == "__main__":
    main()
