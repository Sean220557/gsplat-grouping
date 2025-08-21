# examples/grouping/prepare_priors.py
import argparse, os, glob
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import imageio.v2 as imageio
import cv2

import sys
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES_DIR))
from datasets.colmap import Dataset as ColmapDataset, Parser as ColmapParser


def to_uint16_png(depth_float: np.ndarray) -> np.ndarray:
    d = depth_float.copy()
    d[~np.isfinite(d)] = 0.0
    if d.max() <= 0:
        return (d * 0).astype(np.uint16)
    d = d / (d.max() + 1e-8)
    return (d * 65535.0 + 0.5).astype(np.uint16)


def depth_to_normal(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    H, W = depth.shape
    device = depth.device
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    z = depth.clamp(min=1e-6)
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Pw = torch.stack([X, Y, z], dim=-1)  # [H,W,3]
    dx = Pw[:, 1:, :] - Pw[:, :-1, :]
    dy = Pw[1:, :, :] - Pw[:-1, :, :]
    Hm, Wm = H - 1, W - 1
    vx, vy = dx[:Hm, :Wm, :], dy[:Hm, :Wm, :]
    n = torch.linalg.cross(vx, vy)  # [H-1,W-1,3]
    n = F.normalize(n, dim=-1, eps=1e-6)
    normals = torch.zeros((H, W, 3), device=device, dtype=depth.dtype)
    normals[:Hm, :Wm, :] = n
    normals[Hm:, :Wm, :] = n[-1:, :, :]
    normals[:Hm, Wm:, :] = n[:, -1:, :]
    normals[Hm:, Wm:, :] = n[-1:, -1:, :]
    return normals


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Dataset（拿 K、分辨率）
    parser = ColmapParser(data_dir=args.data_dir, factor=args.data_factor)
    base = ColmapDataset(parser, split="train")

    # 2) 真实文件名列表（**以 images_dir 为准**，保证与 masks/训练完全对齐）
    img_dir = (
        Path(args.images_dir)
        if args.images_dir
        else Path(args.data_dir) / f"images_{args.data_factor}"
    )
    if not img_dir.exists():
        # 退回到 images/
        img_dir = Path(args.data_dir) / "images"
    files = sorted(
        glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png"))
    )
    assert len(files) > 0, f"No images found in {img_dir}"
    if len(files) != len(base):
        print(
            f"[warn] images in {img_dir} = {len(files)} != dataset size {len(base)}; will use min length."
        )
    n = min(len(files), len(base))
    stems = [Path(f).stem for f in files[:n]]

    # 3) 输出目录
    out_depth = Path(args.out_root) / "depth"
    out_depth.mkdir(parents=True, exist_ok=True)
    out_normal = Path(args.out_root) / "normal"
    out_normal.mkdir(parents=True, exist_ok=True)

    # 4) MiDaS(DPT-Large)
    print("loading MiDaS (DPT-Large) …")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    tfm = transforms.dpt_transform

    for i in range(n):
        item = base[i]
        stem = stems[i]

        # H,W,K
        if "height" in item and "width" in item:
            H, W = int(item["height"]), int(item["width"])
        else:
            img = item["image"]
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.shape[-1] in (3, 4):
                    H, W = int(img.shape[0]), int(img.shape[1])
                else:
                    H, W = int(img.shape[1]), int(img.shape[2])
            else:
                H, W = int(img.shape[0]), int(img.shape[1])
        K = item["K"].to(device).float()

        # 读原图（以 images_dir 文件为准）
        im = imageio.imread(files[i])  # HxWx[3|4], uint8
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        if im.shape[2] == 4:
            im = im[..., :3]

        # 5) MiDaS 推理（相对深度），再 resize 回 HxW
        inp = tfm(im).to(device)
        pred = midas(inp).squeeze()  # [h,w] 相对深度（越大越远/近，取决于模型）
        pred = (
            F.interpolate(
                pred[None, None], size=(H, W), mode="bicubic", align_corners=True
            )
            .squeeze()
            .float()
        )

        # 6) 保存深度（npy+png16）
        npy_path = out_depth / f"{stem}.npy"
        png_path = out_depth / f"{stem}.png"
        np.save(npy_path.as_posix(), pred.detach().cpu().numpy().astype(np.float32))
        png16 = to_uint16_png(pred.detach().cpu().numpy())
        imageio.imwrite(png_path.as_posix(), png16)
        # 注意：group_finetune.py 更偏好读取 .npy（浮点），.png 仅作备用/可视化

        # 7) 从深度导法线（用相机 K）
        normal = depth_to_normal(pred, K)  # [H,W,3], [-1,1]
        n_npy = out_normal / f"{stem}.npy"
        n_png = out_normal / f"{stem}.png"
        np.save(n_npy.as_posix(), normal.detach().cpu().numpy().astype(np.float32))
        # 保存一份 0..255 的可视化 PNG
        n_vis = ((normal.detach().cpu().numpy() * 0.5 + 0.5) * 255.0 + 0.5).astype(
            np.uint8
        )
        imageio.imwrite(n_png.as_posix(), n_vis)

        if i % 20 == 0:
            print(f"[{i}/{n}] wrote depth+normal for {stem}")

    print("Done. Prior root:", Path(args.out_root).resolve().as_posix())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument(
        "--images_dir", type=str, default=None, help="显式指定，如 data/.../images_4"
    )
    ap.add_argument(
        "--out_root", required=True, help="输出根目录，例如 data/.../priors"
    )
    args = ap.parse_args()
    run(args)
