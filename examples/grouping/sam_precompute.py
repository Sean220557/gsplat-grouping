import argparse, os, json, glob
from pathlib import Path
import numpy as np
import torch
import cv2

# pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to(device=device)
    amg = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=256,
    )

    out_dir = Path(args.mask_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.images_dir:
        img_dir = Path(args.images_dir)
    else:
        img_root = Path(args.data_dir)
        cand = sorted(
            [p for p in img_root.glob("images_*") if p.is_dir()], key=lambda p: p.name
        )
        img_dir = cand[-1] if cand else img_root / "images"

    jpgs = sorted(glob.glob(str(img_dir / "*.jpg"))) + sorted(
        glob.glob(str(img_dir / "*.png"))
    )

    for i, p in enumerate(jpgs):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ms = amg.generate(img)
        # 过滤/排序：按面积大到小，最多取 K 个实例
        ms = sorted(ms, key=lambda m: m["area"], reverse=True)[: args.max_instances]
        h, w = img.shape[:2]
        label = np.zeros((h, w), dtype=np.uint16)
        for k, m in enumerate(ms, start=1):
            label[m["segmentation"] > 0] = k
        op = out_dir / (Path(p).stem + ".png")
        cv2.imwrite(str(op), label)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(jpgs)}] -> {op}")

    with open(out_dir / "classes.json", "w") as f:
        json.dump({"background": 0}, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir", type=str, required=True, help="例如 data/360_v2/garden"
    )
    ap.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="输出 mask 的目录，例如 data/360_v2/garden/masks",
    )
    ap.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="显式指定 images 目录，如 data/.../images_4",
    )
    ap.add_argument("--sam_model", type=str, default="vit_h")
    ap.add_argument("--sam_ckpt", type=str, required=True)
    ap.add_argument("--max_instances", type=int, default=30)
    args = ap.parse_args()
    run(args)
