# examples/grouping/group_finetune.py
import os, math, argparse, glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gsplat.rendering import rasterization

# 本项目模块
from load_ckpt import load_ckpt
from dataset_adapter import GroupingDataset


# =========================
#     模型组件
# =========================
class SegHead(nn.Module):
    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d, num_classes, bias=True)

    def forward(self, feat_hw_d: torch.Tensor) -> torch.Tensor:
        # 期望 [H,W,D]
        if feat_hw_d.dim() == 4 and feat_hw_d.shape[0] == 1:
            feat_hw_d = feat_hw_d[0]
        if feat_hw_d.dim() == 2:
            feat_hw_d = feat_hw_d.unsqueeze(1)  # 兜底
        H, W, D = feat_hw_d.shape
        logits = self.proj(feat_hw_d.view(-1, D)).view(H, W, -1)  # [H,W,C]
        return logits


# =========================
#     工具函数
# =========================
def to_hwc_uint8(img) -> np.ndarray:
    """将 batch item 的 image 转成 [H,W,3] uint8，用于可视化/重投影。"""
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


def colorize_labels(labels_hw: torch.Tensor) -> np.ndarray:
    """int32 [H,W] → uint8 [H,W,3] 随机调色板（label 0 用深灰）。"""
    H, W = int(labels_hw.shape[0]), int(labels_hw.shape[1])
    n = int(labels_hw.max().item()) + 1
    rng = np.random.RandomState(0)
    cmap = rng.randint(0, 255, size=(max(n, 256), 3), dtype=np.uint8)
    cmap[0] = np.array([40, 40, 40], dtype=np.uint8)
    lab = labels_hw.detach().cpu().numpy().astype(np.int32)
    return cmap[lab]


def _meshgrid_xy(H, W, device):
    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    return x.float(), y.float()


def _ssim(img1, img2, C1=0.01 ** 2, C2=0.03 ** 2):
    """单尺度 SSIM（通道均值），img∈[0,1]，[H,W,3]"""
    t1 = img1.permute(2, 0, 1)[None]
    t2 = img2.permute(2, 0, 1)[None]
    mu1 = F.avg_pool2d(t1, 3, 1, 1)
    mu2 = F.avg_pool2d(t2, 3, 1, 1)
    sigma1 = F.avg_pool2d(t1 * t1, 3, 1, 1) - mu1 * mu1
    sigma2 = F.avg_pool2d(t2 * t2, 3, 1, 1) - mu2 * mu2
    sigma12 = F.avg_pool2d(t1 * t2, 3, 1, 1) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 + sigma2 + C2) + 1e-8
    )
    return ssim_map.mean()


def _warp_ref_to_tgt(
    img_ref: torch.Tensor,  # [H,W,3] in [0,1], device
    D_t: torch.Tensor,  # [H,W]
    K_t: torch.Tensor,
    viewmat_t: torch.Tensor,
    K_r: torch.Tensor,
    viewmat_r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """用目标视角深度把参考图像重投影到目标视角，返回 C_tilde 和有效掩码。"""
    device = img_ref.device
    H, W = int(D_t.shape[0]), int(D_t.shape[1])

    x, y = _meshgrid_xy(H, W, device)
    ones = torch.ones_like(x)
    pix = torch.stack([x, y, ones], dim=-1).view(-1, 3)  # [HW,3]
    Kt_inv = torch.inverse(K_t)

    # 相机坐标方向
    dirs_cam = (Kt_inv @ pix.T).T
    dirs_cam = dirs_cam / (dirs_cam[:, 2:3].clamp(min=1e-6))

    # 世界坐标
    Twc_t = torch.inverse(viewmat_t)  # c2w
    Rwc_t = Twc_t[:3, :3]
    cw_t = Twc_t[:3, 3]
    Xw = cw_t[None, :] + (Rwc_t @ dirs_cam.T).T * D_t.view(-1, 1)

    # 到参考相机坐标：x_r = R_cw * x_w + t_cw
    Rcw_r = viewmat_r[:3, :3]
    tcw_r = viewmat_r[:3, 3]
    Xr = (Rcw_r @ Xw.T).T + tcw_r[None, :]
    zr = Xr[:, 2].clamp(min=1e-6)
    proj = (K_r @ (Xr / zr[:, None]).T).T
    u, v = proj[:, 0], proj[:, 1]

    u_norm = (u / (W - 1) * 2 - 1).clamp(-1, 1)
    v_norm = (v / (H - 1) * 2 - 1).clamp(-1, 1)
    grid = torch.stack([u_norm, v_norm], dim=-1).view(1, H, W, 2)

    ref = img_ref.permute(2, 0, 1)[None]  # [1,3,H,W]
    Ct = F.grid_sample(
        ref, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0].permute(1, 2, 0)

    valid = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1) & torch.isfinite(zr)
    return Ct, valid.view(H, W)


def _depth_to_normal(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    从深度图估计法线（相机坐标系），depth:[H,W]，K:[3,3] -> normals:[H,W,3] in [-1,1]
    近似：邻域有限差分 + 反投影
    """
    H, W = int(depth.shape[0]), int(depth.shape[1])
    device = depth.device
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 计算每个像素的 3D 坐标
    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    z = depth.clamp(min=1e-6)
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Pw = torch.stack([X, Y, z], dim=-1)  # [H,W,3] 相机坐标

    # 邻域差分
    dx = Pw[:, 1:, :] - Pw[:, :-1, :]
    dy = Pw[1:, :, :] - Pw[:-1, :, :]

    # 对齐到 H-1, W-1
    Hm, Wm = H - 1, W - 1
    vx = dx[:Hm, :Wm, :]
    vy = dy[:Hm, :Wm, :]

    n = torch.linalg.cross(vx, vy)  # [H-1,W-1,3]
    n = F.normalize(n, dim=-1, eps=1e-6)

    # 填回到原分辨率（边界复制）
    normals = torch.zeros((H, W, 3), device=device, dtype=depth.dtype)
    normals[:Hm, :Wm, :] = n
    normals[Hm:, :Wm, :] = n[-1:, :, :]
    normals[:Hm, Wm:, :] = n[:, -1:, :]
    normals[Hm:, Wm:, :] = n[-1:, -1:, :]

    return normals


def _load_depth_prior(prior_dir: Optional[str], stem: str, H: int, W: int, device) -> Optional[torch.Tensor]:
    if not prior_dir:
        return None
    p_npy = Path(prior_dir, "depth", f"{stem}.npy")
    p_png = Path(prior_dir, "depth", f"{stem}.png")
    if p_npy.exists():
        d = torch.from_numpy(np.load(p_npy.as_posix())).float()
    elif p_png.exists():
        import imageio.v2 as imageio
        d = torch.from_numpy(imageio.imread(p_png.as_posix()).astype(np.float32))
        if d.max() > 1.0:
            d = d / 1000.0  # 粗略假设毫米
    else:
        return None
    if d.ndim == 3:
        d = d[..., 0]
    d = d.to(device)
    if d.shape[0] != H or d.shape[1] != W:
        d = F.interpolate(d[None, None], size=(H, W), mode="nearest").squeeze(0).squeeze(0)
    return d


def _load_normal_prior(prior_dir: Optional[str], stem: str, H: int, W: int, device) -> Optional[torch.Tensor]:
    if not prior_dir:
        return None
    p_npy = Path(prior_dir, "normal", f"{stem}.npy")
    p_png = Path(prior_dir, "normal", f"{stem}.png")
    if p_npy.exists():
        n = torch.from_numpy(np.load(p_npy.as_posix()).astype(np.float32))  # HxWx3
    elif p_png.exists():
        import imageio.v2 as imageio
        n = torch.from_numpy(imageio.imread(p_png.as_posix()).astype(np.float32))  # 0..255
        if n.ndim == 2:
            n = n[..., None].repeat(3, axis=-1)
        n = n / 255.0 * 2.0 - 1.0
    else:
        return None
    n = n.to(device)
    if n.shape[0] != H or n.shape[1] != W:
        n = (
            F.interpolate(n.permute(2, 0, 1)[None], size=(H, W), mode="nearest")
            .squeeze(0)
            .permute(1, 2, 0)
        )
    n = F.normalize(n, dim=-1, eps=1e-6)
    return n


def _align_depth_scale_shift(D_render: torch.Tensor, D_prior: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """用最小二乘在有效像素上拟合 a*D_prior + b ≈ D_render，返回对齐后的 prior。"""
    v = valid & torch.isfinite(D_render) & torch.isfinite(D_prior)
    if v.sum() < 100:
        return D_prior
    x = torch.stack([D_prior[v], torch.ones_like(D_prior[v])], dim=1)  # [N,2]
    y = D_render[v][:, None]  # [N,1]
    sol = torch.linalg.lstsq(x, y).solution  # [2,1]
    a, b = sol[0, 0], sol[1, 0]
    return a * D_prior + b


# =========================
#     渲染特征 + 深度
# =========================
@torch.no_grad()
def _rasterize_once(
    means, quats, scales, opacities, colors, viewmat, K, width, height, camera_model="pinhole", render_mode=None
):
    # 优先尝试带期望深度的模式；若不支持 render_mode，可省略该参数
    kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None, ...],
        Ks=K[None, ...],
        width=width,
        height=height,
        sh_degree=None,
        packed=True,
        rasterize_mode="classic",
        camera_model=camera_model,
    )
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    render, alphas, _ = rasterization(**kwargs)
    return render[0], (alphas[0, :, :, 0] if alphas is not None else None)  # [H,W,C], [H,W] or None


def render_id_feature_once(
    means, quats, scales, opacities, id_codes, viewmat, K, width: int, height: int, ssaa: int = 1, camera_model="pinhole"
):
    ss = max(1, int(ssaa))
    Hi, Wi = int(height * ss), int(width * ss)

    # 一次渲染拿到 [H,W,D+1]：D维特征 + 期望深度（若支持），以及 alpha
    try:
        render_hi, alpha_hi = _rasterize_once(
            means, quats, scales, opacities, id_codes, viewmat, K, Wi, Hi, camera_model, render_mode="RGB+ED"
        )
        feat_hi = render_hi[:, :, :-1]
        depth_hi = render_hi[:, :, -1]
    except TypeError:
        # 兼容没有 render_mode 的 gsplat：只渲染特征；深度设为全 0（几何项将被弱化）
        render_hi, alpha_hi = _rasterize_once(
            means, quats, scales, opacities, id_codes, viewmat, K, Wi, Hi, camera_model, render_mode=None
        )
        feat_hi = render_hi
        depth_hi = torch.zeros((Hi, Wi), device=feat_hi.device, dtype=feat_hi.dtype)

    if ss > 1:
        feat = (
            F.avg_pool2d(feat_hi.permute(2, 0, 1)[None], kernel_size=ss, stride=ss)
            .squeeze(0)
            .permute(1, 2, 0)
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
    return feat, depth, alpha  # [H,W,D], [H,W], [H,W]


# =========================
#     评测（硬标签）
# =========================
@torch.no_grad()
def do_eval_logits(
    ds: GroupingDataset,
    means, quats, scales, opacities, id_codes, head: SegHead, device,
    out_dir: str, camera_model: str, indices: List[int], overlay: bool, alpha: float, ssaa: int
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    head.eval()
    import imageio.v2 as imageio

    for idx in indices:
        if idx < 0 or idx >= len(ds):
            print(f"[eval] skip idx={idx} out of range")
            continue
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
            means, quats, scales, opacities, id_codes, viewmat, K, W, H, ssaa=ssaa, camera_model=camera_model
        )
        logits = head(feat)
        pred = logits.argmax(dim=-1)
        color = colorize_labels(pred)

        if overlay:
            rgb = to_hwc_uint8(img)
            if rgb.shape[:2] != color.shape[:2]:
                try:
                    import cv2
                    color = cv2.resize(color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    # 备用：用 torch 最近邻
                    t = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(0).float()
                    t = F.interpolate(t, size=(rgb.shape[0], rgb.shape[1]), mode="nearest")
                    color = t.squeeze(0).permute(1, 2, 0).byte().numpy()
            blend = (alpha * color + (1 - alpha) * rgb).astype(np.uint8)
            out_img = blend
        else:
            out_img = color

        name = Path(item.get("path", f"{idx:06d}")).stem
        out_path = Path(out_dir) / f"{name}.png"
        imageio.imwrite(out_path.as_posix(), out_img)
        print(f"[eval] wrote: {out_path.as_posix()}")


def parse_indices(s: str) -> List[int]:
    if not s:
        return []
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
    return out


# =========================
#     主训练流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    # 数据 / ckpt
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--mask_dir", type=str, required=True)
    ap.add_argument("--images_dir", type=str, default=None, help="与 masks 一一对应的 images 目录（如 images_4）")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--result_dir", type=str, default="results/grouping_run")

    # 模型/优化
    ap.add_argument("--id_dim", type=int, default=32)
    ap.add_argument("--max_classes", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--ignore_index", type=int, default=0)
    ap.add_argument("--camera_model", type=str, default="pinhole")

    # 稳边界调参
    ap.add_argument("--lr_id", type=float, default=3e-3)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--tv_weight", type=float, default=5e-4)
    ap.add_argument("--edge_weight", type=float, default=1.0)
    ap.add_argument("--min_fg_ratio", type=float, default=0.01)
    ap.add_argument("--ssaa", type=int, default=1)

    # 评测/渲染（硬标签）
    ap.add_argument("--eval_mode", type=str, default="logits", choices=["none", "logits"])
    ap.add_argument("--eval_indices", type=str, default="0,10,20", help="逗号/区间，train 索引用于评测渲染")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_overlay", action="store_true")
    ap.add_argument("--overlay_alpha", type=float, default=0.35)

    # --- GOC Geometry Regularization ---
    ap.add_argument("--geo_enable", action="store_true", help="启用 GOC 4.2 的几何正则（Ld/Ldn/Lpho/Lgeo）")
    ap.add_argument("--prior_depth_dir", type=str, default=None,
                    help="深度先验目录：depth/{stem}.npy|png，与 images 文件名对齐")
    ap.add_argument("--prior_normal_dir", type=str, default=None,
                    help="法线先验目录：normal/{stem}.npy|png，PNG 将被映射到[-1,1]")
    ap.add_argument("--lambda_d", type=float, default=0.3)
    ap.add_argument("--lambda_dn", type=float, default=0.1)
    ap.add_argument("--lambda_pho", type=float, default=0.3)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--pho_ssim_lambda", type=float, default=0.85)
    ap.add_argument("--pho_pair_stride", type=int, default=1)

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据
    ds = GroupingDataset(
        data_dir=args.data_dir,
        factor=args.data_factor,
        split="train",
        patch_size=None,
        mask_dir=args.mask_dir,
        images_dir=args.images_dir,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    # 读取 gsplat ckpt（按官方激活）
    base = load_ckpt(args.ckpt, device=device)
    means, quats, scales = [base[k].to(device) for k in ("means", "quats", "scales")]
    opacities = base["opacities"].to(device)

    N = means.shape[0]
    id_codes = nn.Parameter(F.normalize(torch.randn(N, args.id_dim, device=device), dim=-1))
    head = SegHead(args.id_dim, args.max_classes).to(device)

    optim = torch.optim.AdamW(
        [{"params": [id_codes], "lr": args.lr_id}, {"params": head.parameters(), "lr": args.lr_head}],
        weight_decay=1e-4,
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    os.makedirs(args.result_dir, exist_ok=True)
    global_step = 0

    for ep in range(args.epochs):
        for it, batch in enumerate(dl):
            # ---- 相机/图像/标签 ----
            img = batch["image"][0]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            img = img.to(device).float()

            viewmat = batch["viewmat"][0].to(device).float()
            K = batch["K"][0].to(device).float()

            mask0 = batch["mask"][0]
            gt = mask0.to(device).long() if isinstance(mask0, torch.Tensor) else torch.from_numpy(mask0).to(device).long()
            H, W = int(gt.shape[0]), int(gt.shape[1])

            # ---- 跳过无前景样本 ----
            valid_pix = (gt != args.ignore_index)
            fg_ratio = valid_pix.float().mean().item()
            if fg_ratio < args.min_fg_ratio:
                continue

            # ---- 渲染特征 + 期望深度 + alpha ----
            feat, depth, alpha = render_id_feature_once(
                means, quats, scales, opacities, id_codes, viewmat, K, W, H, ssaa=args.ssaa, camera_model=args.camera_model
            )  # [H,W,D], [H,W], [H,W]

            # ---- 分类头 + 边界权重 CE ----
            num_cls = min(int(gt.max().item()) + 1, args.max_classes)
            logits = head(feat)[..., :num_cls]

            with torch.no_grad():
                # 灰度提边
                if img.dim() == 3 and img.shape[-1] in (3, 4):  # HWC
                    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).unsqueeze(0).unsqueeze(0)
                elif img.dim() == 3 and img.shape[0] in (1, 3, 4):  # CHW
                    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0).unsqueeze(0)
                else:
                    gray = (gt.float().unsqueeze(0).unsqueeze(0) > 0).float()
                sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=gray.device, dtype=gray.dtype).view(
                    1, 1, 3, 3
                )
                sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=gray.device, dtype=gray.dtype).view(
                    1, 1, 3, 3
                )
                gx = F.conv2d(gray, sobel_x, padding=1)
                gy = F.conv2d(gray, sobel_y, padding=1)
                edges = torch.sqrt(gx ** 2 + gy ** 2).squeeze()
                edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
                weight_map = (1.0 + args.edge_weight * edges) * valid_pix.float() * (alpha > 0.05).float()

            logits_flat = logits.view(-1, logits.shape[-1])
            gt_flat = gt.view(-1)
            weight_flat = weight_map.view(-1)
            ce_pix = F.cross_entropy(logits_flat, gt_flat, reduction="none")
            loss_ce = (ce_pix * weight_flat).sum() / (weight_flat.sum() + 1e-8)

            # ---- TV 正则（在 softmax 概率上）----
            prob = F.softmax(logits, dim=-1)
            dx = prob[1:, :, :] - prob[:-1, :, :]
            dy = prob[:, 1:, :] - prob[:, :-1, :]
            loss_tv = args.tv_weight * (dx.abs().mean() + dy.abs().mean())

            # ---- 轻 L2 正则 ----
            loss_reg = (id_codes ** 2).mean() * 1e-4

            # =============================
            #   GOC 4.2 Geometry Regularization
            # =============================
            loss_d = torch.tensor(0.0, device=device)
            loss_dn = torch.tensor(0.0, device=device)
            loss_pho = torch.tensor(0.0, device=device)
            loss_geo = torch.tensor(0.0, device=device)

            if args.geo_enable:
                # 当前帧文件名 stem（用于加载先验）
                if "path" in batch:
                    stem = Path(batch["path"][0]).stem
                else:
                    stem = f"{it:06d}"

                # (a) 深度先验：对齐 scale+shift 后 L1
                D_hat = _load_depth_prior(args.prior_depth_dir, stem, H, W, device)
                if D_hat is not None:
                    v = (alpha > 0.05) & torch.isfinite(depth) & torch.isfinite(D_hat)
                    if v.any():
                        D_hat_aligned = _align_depth_scale_shift(depth, D_hat, v)
                        loss_d = torch.abs(depth - D_hat_aligned)[v].mean()

                # (b) 法线先验：从深度估计法线，与先验作 1 - dot，alpha 加权
                N_d = _depth_to_normal(depth, K)  # [H,W,3]
                N_hat = _load_normal_prior(args.prior_normal_dir, stem, H, W, device)
                if N_hat is not None:
                    dot = (F.normalize(N_d, dim=-1) * F.normalize(N_hat, dim=-1)).sum(-1).clamp(-1, 1)
                    loss_dn = (alpha * (1.0 - dot)).mean()

                # (c) 光度重投影（目标 ← 参考）
                ref_idx = (
                    (batch.get("index", [it])[0] if isinstance(batch.get("index", [it]), list) else it)
                    + int(args.pho_pair_stride)
                ) % len(ds)
                item_ref = ds[ref_idx]
                viewmat_r = item_ref["viewmat"].to(device).float()
                K_r = item_ref["K"].to(device).float()
                img_r = item_ref["image"]
                rgb_r = torch.from_numpy(to_hwc_uint8(img_r).astype(np.float32) / 255.0).to(device)
                rgb_t = torch.from_numpy(to_hwc_uint8(img).astype(np.float32) / 255.0).to(device)

                C_tilde, vld = _warp_ref_to_tgt(rgb_r, depth, K, viewmat, K_r, viewmat_r)
                lamb = float(args.pho_ssim_lambda)
                ssim_val = _ssim(rgb_t, C_tilde)
                l1 = torch.abs(rgb_t - C_tilde)[vld].mean() if vld.any() else torch.tensor(0.0, device=device)
                loss_pho = (1.0 - ssim_val) * lamb / 2.0 + (1.0 - lamb) * l1

                # (d) 几何重投影一致性（目标→参考 单向）
                # 渲染参考视角深度
                _, D_ref, _ = render_id_feature_once(
                    means, quats, scales, opacities, id_codes, viewmat_r, K_r, W, H, ssaa=1, camera_model=args.camera_model
                )

                # 用目标深度生成世界点 → 参考像素上采样 D_ref 比较
                x, y = _meshgrid_xy(H, W, device)
                ones = torch.ones_like(x)
                pix = torch.stack([x, y, ones], dim=-1).view(-1, 3)
                Kt_inv = torch.inverse(K)
                dirs_cam = (Kt_inv @ pix.T).T
                dirs_cam = dirs_cam / (dirs_cam[:, 2:3].clamp(min=1e-6))
                Twc_t = torch.inverse(viewmat)
                Rwc_t = Twc_t[:3, :3]
                cw_t = Twc_t[:3, 3]
                Xw_t = cw_t[None, :] + (Rwc_t @ dirs_cam.T).T * depth.view(-1, 1)

                Rcw_r = viewmat_r[:3, :3]
                tcw_r = viewmat_r[:3, 3]
                Xr = (Rcw_r @ Xw_t.T).T + tcw_r[None, :]
                zr = Xr[:, 2].clamp(min=1e-6)
                proj = (K_r @ (Xr / zr[:, None]).T).T
                u = proj[:, 0].view(H, W)
                v = proj[:, 1].view(H, W)
                u_norm = (u / (W - 1) * 2 - 1).clamp(-1, 1)
                v_norm = (v / (H - 1) * 2 - 1).clamp(-1, 1)
                grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
                D_ref_smpl = F.grid_sample(
                    D_ref[None, None], grid, mode="nearest", padding_mode="zeros", align_corners=True
                )[0, 0]
                mask = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
                loss_geo = (
                    torch.abs(D_ref_smpl[mask] - zr.view(H, W)[mask]).mean()
                    if mask.any()
                    else torch.tensor(0.0, device=device)
                )

            # ---- 总损失 ----
            loss = (
                loss_ce
                + loss_tv
                + loss_reg
                + args.lambda_d * loss_d
                + args.lambda_dn * loss_dn
                + args.lambda_pho * loss_pho
                + args.lambda_geo * loss_geo
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + [id_codes], max_norm=args.clip_grad)
            optim.step()
            with torch.no_grad():
                id_codes.data = F.normalize(id_codes.data, dim=-1)
            scheduler.step()

            if global_step % 50 == 0:
                print(
                    f"[ep {ep} it {it}] "
                    f"loss={loss.item():.4f} "
                    f"(ce={loss_ce.item():.4f}, tv={loss_tv.item():.4f}, reg={loss_reg.item():.6f}, "
                    f"d={float(loss_d):.4f}, dn={float(loss_dn):.4f}, pho={float(loss_pho):.4f}, geo={float(loss_geo):.4f})"
                )
            global_step += 1

        # ---- 保存 ckpt ----
        ckpt_path = os.path.join(args.result_dir, f"grouping_ep{ep:02d}.pt")
        torch.save(
            {
                "splats": {
                    "means": means.detach().cpu(),
                    "quats": quats.detach().cpu(),
                    "scales": torch.log(scales.detach().cpu()),  # 保存未激活量
                    "opacities": torch.logit(opacities.detach().cpu().clamp(1e-6, 1 - 1e-6)),
                    "sh0": base["sh0"].detach().cpu(),
                    "shN": base["shN"].detach().cpu(),
                },
                "grouping": {
                    "id_codes": id_codes.detach().cpu(),
                    "id_dim": args.id_dim,
                    "seg_head": head.state_dict(),
                    "num_classes": args.max_classes,
                },
                "meta": {"source_ckpt": args.ckpt},
                "step": base.get("step", None),
            },
            ckpt_path,
        )
        print(f"saved: {ckpt_path}")

        # ---- 评测渲染：硬标签 logits ----
        if args.eval_mode == "logits" and ((ep + 1) % args.eval_every == 0):
            do_eval_logits(
                ds,
                means,
                quats,
                scales,
                opacities,
                id_codes,
                head,
                device,
                out_dir=os.path.join(args.result_dir, f"eval_ep{ep:02d}"),
                camera_model=args.camera_model,
                indices=parse_indices(args.eval_indices),
                overlay=args.eval_overlay,
                alpha=args.overlay_alpha,
                ssaa=args.ssaa,
            )


if __name__ == "__main__":
    main()
