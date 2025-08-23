import os, math, argparse, glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.rendering import rasterization
from dataset_adapter import GroupingDataset


def parse_indices(s: str, n_total: int) -> List[int]:
    if s in (None, "", "all", "*"):
        return list(range(n_total))
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            a, b = int(a), int(b)
            out.extend(list(range(min(a, b), max(a, b) + 1)))
        else:
            out.append(int(tok))
    # 去重/裁剪
    out = sorted(set([i for i in out if 0 <= i < n_total]))
    return out


def to_hwc_uint8(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        x = img
        if x.dim() == 3 and x.shape[0] in (1, 3, 4):  # CHW
            x = x.permute(1, 2, 0)
        x = x[..., :3].clamp(0, 1).detach().cpu().numpy()
        return (x * 255.0 + 0.5).astype(np.uint8)
    elif isinstance(img, np.ndarray):
        x = img[..., :3]
        if x.dtype != np.uint8:
            x = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        return x
    raise ValueError("Unsupported image type")


def colorize_labels(labels_hw: torch.Tensor, seed: int = 0) -> np.ndarray:
    H, W = int(labels_hw.shape[0]), int(labels_hw.shape[1])
    n = int(labels_hw.max().item()) + 1
    rng = np.random.RandomState(seed)
    cmap = rng.randint(0, 255, size=(max(n, 256), 3), dtype=np.uint8)
    cmap[0] = np.array([40, 40, 40], dtype=np.uint8)
    lab = labels_hw.detach().cpu().numpy().astype(np.int32)
    return cmap[lab]


class SegHead(nn.Module):
    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d, num_classes, bias=True)

    def forward(self, feat_hw_d: torch.Tensor) -> torch.Tensor:
        if feat_hw_d.dim() == 4 and feat_hw_d.shape[0] == 1:
            feat_hw_d = feat_hw_d[0]
        H, W, D = feat_hw_d.shape
        logits = self.proj(feat_hw_d.view(-1, D)).view(H, W, -1)  # [H,W,C]
        return logits


@torch.no_grad()
def _rasterize_once(
    means, quats, scales, opacities, colors, viewmat, K, width, height, camera_model="pinhole", render_mode=None
):
    kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None, ...],
        Ks=K[None, ...],
        width=int(width),
        height=int(height),
        sh_degree=None,
        packed=True,
        rasterize_mode="classic",
        camera_model=camera_model,
    )
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    render, alphas, _ = rasterization(**kwargs)
    return render[0], (alphas[0, :, :, 0] if alphas is not None else None)  # [H,W,C], [H,W] or None


@torch.no_grad()
def render_id_feature_once(
    means, quats, scales, opacities, id_codes, viewmat, K, width: int, height: int, ssaa: int = 1, camera_model="pinhole"
):
    ss = max(1, int(ssaa))
    Hi, Wi = int(height * ss), int(width * ss)

    try:
        render_hi, alpha_hi = _rasterize_once(
            means, quats, scales, opacities, id_codes, viewmat, K, Wi, Hi, camera_model, render_mode="RGB+ED"
        )
        feat_hi = render_hi[:, :, :-1]
        depth_hi = render_hi[:, :, -1]
    except TypeError:
        means_cam = (viewmat[:3, :3] @ means.T + viewmat[:3, 3:4]).T
        z_cam = means_cam[:, 2:3]
        colors_aug = torch.cat([id_codes, z_cam], dim=1)
        render_hi, alpha_hi = _rasterize_once(
            means, quats, scales, opacities, colors_aug, viewmat, K, Wi, Hi, camera_model, render_mode=None
        )
        feat_hi = render_hi[:, :, :-1]
        depth_hi = render_hi[:, :, -1]

    if ss > 1:
        feat = (
            F.avg_pool2d(feat_hi.permute(2, 0, 1)[None], kernel_size=ss, stride=ss)
            .squeeze(0).permute(1, 2, 0)
        )
        depth = F.avg_pool2d(depth_hi[None, None], kernel_size=ss, stride=ss).squeeze(0).squeeze(0)
        alpha = (
            F.avg_pool2d(alpha_hi[None, None], kernel_size=ss, stride=ss).squeeze(0).squeeze(0)
            if alpha_hi is not None
            else torch.zeros((height, width), device=feat_hi.device, dtype=feat_hi.dtype)
        )
    else:
        feat, depth, alpha = feat_hi, depth_hi, (alpha_hi if alpha_hi is not None else torch.zeros_like(depth_hi))

    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = torch.clamp(alpha, 0.0, 1.0)
    return feat, depth, alpha


def main():
    ap = argparse.ArgumentParser("Render grouping results (hard labels)")
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--images_dir", type=str, default=None, help="与训练一致的图像目录（如 images_4）")
    ap.add_argument("--mask_dir", type=str, default=None, help="可选，仅用于从数据集取 K/viewmat 的适配器")
    ap.add_argument("--ckpt", required=True, type=str, help="训练导出的 grouping_epXX.pt")
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--indices", type=str, default="all", help="如 0-160 或 0,10,20，默认全量")
    ap.add_argument("--mode", type=str, default="logits", choices=["logits"], help="此脚本专注硬标签展示")
    ap.add_argument("--overlay", action="store_true", help="是否叠加原图做可视化")
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay 透明度（标签权重）")
    ap.add_argument("--ssaa", type=int, default=1)
    ap.add_argument("--camera_model", type=str, default="pinhole")
    ap.add_argument("--palette_seed", type=int, default=0, help="着色随机种（保证跨帧一致）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ds = GroupingDataset(
        data_dir=args.data_dir,
        factor=args.data_factor,
        split="train",
        patch_size=None,
        mask_dir=args.mask_dir,
        images_dir=args.images_dir,
    )

    render_ids = parse_indices(args.indices, len(ds))
    print(f"[render] total={len(ds)}, will render {len(render_ids)} frames")

    ckpt = torch.load(args.ckpt, map_location=device)
    spl = ckpt["splats"]
    grp = ckpt["grouping"]

    means = spl["means"].to(device).float()                 # [N,3]
    quats = spl["quats"].to(device).float()                 # [N,4]
    scales = torch.exp(spl["scales"].to(device).float())    # [N,3]
    opacities = torch.sigmoid(spl["opacities"].to(device).float())  # [N,1]
    id_codes = grp["id_codes"].to(device).float()           # [N,D]
    id_dim = int(grp.get("id_dim", id_codes.shape[1]))
    max_classes = int(grp.get("num_classes", 64))
    global_num_classes = int(grp.get("global_num_classes", max_classes))

    head = SegHead(id_dim, max_classes).to(device)
    head.load_state_dict(grp["seg_head"])
    head.eval()

    import imageio.v2 as imageio
    try:
        import cv2
        _has_cv2 = True
    except Exception:
        _has_cv2 = False

    with torch.no_grad():
        for idx in render_ids:
            item = ds[idx]
            viewmat = item["viewmat"].to(device).float()
            K = item["K"].to(device).float()
            img = item["image"]
            if isinstance(img, np.ndarray):
                img_t = torch.from_numpy(img).to(device).float()
            else:
                img_t = img.to(device).float()

            if "height" in item and "width" in item:
                H, W = int(item["height"]), int(item["width"])
            else:
                if img_t.dim() == 3 and img_t.shape[-1] in (3, 4):
                    H, W = int(img_t.shape[0]), int(img_t.shape[1])
                else:
                    H, W = int(img_t.shape[1]), int(img_t.shape[2])

            feat, _, _ = render_id_feature_once(
                means, quats, scales, opacities, id_codes, viewmat, K, W, H, ssaa=args.ssaa, camera_model=args.camera_model
            )
            logits = head(feat)[..., :global_num_classes]
            pred = logits.argmax(dim=-1)  # [H,W]
            color = colorize_labels(pred, seed=args.palette_seed)

            if args.overlay:
                rgb = to_hwc_uint8(img)
                if rgb.shape[:2] != color.shape[:2]:
                    if _has_cv2:
                        color = cv2.resize(color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        t = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(0).float()
                        t = F.interpolate(t, size=(rgb.shape[0], rgb.shape[1]), mode="nearest")
                        color = t.squeeze(0).permute(1, 2, 0).byte().numpy()
                out = (args.alpha * color + (1.0 - args.alpha) * rgb).astype(np.uint8)
            else:
                out = color

            name = Path(item.get("path", f"{idx:06d}")).stem
            out_path = Path(args.out_dir, f"{name}.png")
            imageio.imwrite(out_path.as_posix(), out)
            print(f"[write] {out_path.as_posix()}")

    print("[done] all frames rendered.")


if __name__ == "__main__":
    main()
