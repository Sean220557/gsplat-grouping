# examples/grouping/group_finetune.py
# Full training script for "Gaussian Grouping on gsplat"
# - Identity encodings per Gaussian -> render to 2D -> 1x1 head -> CE
# - Optional GOC-like geometry regularization (depth/normal/photometric/geo)
# - 3D class-consistency (KNN) regularization akin to gaussian-grouping's loss_cls_3d
# - Visibility gating (alpha>thresh), TV+edge, entropy & class-balance, feature diversity
# - Alternating freeze (head/id) to force features learning

import os, math, glob, json, random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# gsplat
from gsplat.rendering import rasterization

# local project
from load_ckpt import load_ckpt
from dataset_adapter import GroupingDataset


# -----------------------------
# utilities
# -----------------------------
def to_hwc_uint8(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        x = img
        if x.dim() == 3 and x.shape[0] in (1, 3, 4):  # CHW -> HWC
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
    H, W = int(labels_hw.shape[0]), int(labels_hw.shape[1])
    n = int(labels_hw.max().item()) + 1
    rng = np.random.RandomState(0)
    cmap = rng.randint(0, 255, size=(max(n, 256), 3), dtype=np.uint8)
    cmap[0] = np.array([40, 40, 40], dtype=np.uint8)
    lab = labels_hw.detach().cpu().numpy().astype(np.int32)
    return cmap[lab]

def meshgrid_xy(H, W, device):
    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    return x.float(), y.float()

# -----------------------------
# tiny SSIM (for photometric optional)
# -----------------------------
def _ssim(img1, img2, C1=0.01 ** 2, C2=0.03 ** 2):
    t1 = img1.permute(2, 0, 1)[None]
    t2 = img2.permute(2, 0, 1)[None]
    mu1 = F.avg_pool2d(t1, 3, 1, 1); mu2 = F.avg_pool2d(t2, 3, 1, 1)
    sigma1 = F.avg_pool2d(t1*t1, 3, 1, 1) - mu1*mu1
    sigma2 = F.avg_pool2d(t2*t2, 3, 1, 1) - mu2*mu2
    sigma12 = F.avg_pool2d(t1*t2, 3, 1, 1) - mu1*mu2
    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / (
        (mu1*mu1 + mu2*mu2 + C1) * (sigma1 + sigma2 + C2) + 1e-8
    )
    return ssim_map.mean()

# -----------------------------
# depth->normal (camera coords)
# -----------------------------
def depth_to_normal(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    H, W = int(depth.shape[0]), int(depth.shape[1])
    device = depth.device
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    z = depth.clamp(min=1e-6)
    X = (x - cx) * z / fx; Y = (y - cy) * z / fy
    Pw = torch.stack([X, Y, z], dim=-1)
    dx = Pw[:, 1:, :] - Pw[:, :-1, :]; dy = Pw[1:, :, :] - Pw[:-1, :, :]
    vx = dx[:-1, :, :]; vy = dy[:, :-1, :]
    n = torch.linalg.cross(vx, vy, dim=-1)
    n = F.normalize(n, dim=-1, eps=1e-6)
    normals = torch.zeros((H, W, 3), device=device, dtype=depth.dtype)
    normals[:-1, :-1, :] = n
    normals[-1, :-1, :] = n[-1, :, :]
    normals[:-1, -1, :] = n[:, -1, :]
    normals[-1, -1, :] = n[-1, -1, :]
    return normals

# -----------------------------
# gsplat raster wrapper
# -----------------------------
def _raster_once(means, quats, scales, opacities, colors, viewmat, K, W, H, camera_model="pinhole", render_mode="RGB+ED"):
    kwargs = dict(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmat[None], Ks=K[None], width=W, height=H,
        sh_degree=None, packed=True, rasterize_mode="classic", camera_model=camera_model
    )
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    render, alphas, _ = rasterization(**kwargs)
    return render[0], (alphas[0, :, :, 0] if alphas is not None else None)

def render_feat_depth_alpha(
    means, quats, scales, opacities, id_as_color, viewmat, K, W, H, ssaa=1, camera_model="pinhole"
):
    ss = max(1, int(ssaa))
    Hi, Wi = int(H*ss), int(W*ss)
    try:
        out, a_hi = _raster_once(means, quats, scales, opacities, id_as_color, viewmat, K, Wi, Hi, camera_model, "RGB+ED")
        feat_hi = out[:, :, :-1]; depth_hi = out[:, :, -1]
    except TypeError:
        means_cam = (viewmat[:3, :3] @ means.T + viewmat[:3, 3:4]).T
        z_cam = means_cam[:, 2:3]
        col = torch.cat([id_as_color, z_cam], dim=1)
        out, a_hi = _raster_once(means, quats, scales, opacities, col, viewmat, K, Wi, Hi, camera_model, None)
        feat_hi = out[:, :, :-1]; depth_hi = out[:, :, -1]

    if ss > 1:
        feat = F.avg_pool2d(feat_hi.permute(2,0,1)[None], kernel_size=ss, stride=ss).squeeze(0).permute(1,2,0)
        depth = F.avg_pool2d(depth_hi[None,None], kernel_size=ss, stride=ss).squeeze(0).squeeze(0)
        alpha = F.avg_pool2d(a_hi[None,None], kernel_size=ss, stride=ss).squeeze(0).squeeze(0) if a_hi is not None else torch.zeros((H,W), device=feat.device, dtype=feat.dtype)
    else:
        feat, depth = feat_hi, depth_hi
        alpha = a_hi if a_hi is not None else torch.zeros_like(depth_hi)
    return torch.nan_to_num(feat), torch.nan_to_num(depth), alpha.clamp(0,1)

# -----------------------------
# tiny 1x1 head
# -----------------------------
class SegHead(nn.Module):
    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d, num_classes, bias=True)
    def forward(self, feat_hw_d: torch.Tensor) -> torch.Tensor:
        if feat_hw_d.dim() == 4 and feat_hw_d.shape[0] == 1: feat_hw_d = feat_hw_d[0]
        H, W, D = feat_hw_d.shape
        return self.proj(feat_hw_d.reshape(-1, D)).view(H, W, -1)

# -----------------------------
# KNN 3D consistency (loss_cls_3d style)
# Encourage per-Gaussian class probs to be smooth in 3D neighbourhood
# -----------------------------
def knn_class_consistency(means_xyz, id_codes, head,
                          k=5, lam=2.0, max_points=300000, sample_size=1000,
                          chunk=50000):  # ☆ 新增：分块大小
    device = means_xyz.device
    N = means_xyz.shape[0]
    idx = torch.randperm(N, device=device)[:min(sample_size, N)]
    X = means_xyz[idx]  # [M,3]
    M = X.shape[0]

    # 运行中的 top-k（距离/索引）
    best_d = torch.full((M, k), float("inf"), device=device)
    best_i = torch.full((M, k), -1, device=device, dtype=torch.long)

    # 分块扫描所有高斯
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        block = means_xyz[s:e]                            # [B,3]
        d2 = torch.cdist(X, block, p=2)                   # [M,B]
        cand_d = torch.cat([best_d, d2], dim=1)           # [M,k+B]
        cand_i = torch.cat([best_i, torch.arange(s, e, device=device)[None].repeat(M, 1)], dim=1)
        # 取更小的 k 个
        val, ind = cand_d.topk(k=k, largest=False, dim=1)
        best_d = val
        best_i = cand_i.gather(1, ind)

    # 用 head 计算被采样点与邻居的概率
    with torch.no_grad():
        logits_c = head(id_codes[idx][None, None, :]).squeeze(0).squeeze(0)   # [M,C]
        prob_c = F.softmax(logits_c, dim=-1)
        neigh = id_codes[best_i.reshape(-1)]
        logits_n = head(neigh[None, None, :]).squeeze(0).squeeze(0)            # [M*k,C]
        prob_n = F.softmax(logits_n, dim=-1).reshape(M, k, -1).mean(1)         # [M,C]

    kl = (prob_c * (prob_c.add(1e-8).log() - prob_n.add(1e-8).log())).sum(-1).mean()
    return lam * kl


# -----------------------------
# eval: logits->hard labels
# -----------------------------
@torch.no_grad()
def eval_logits(ds: GroupingDataset, means, quats, scales, opacities, id_codes, head, device, out_dir: str,
                camera_model: str, indices: List[int], overlay: bool, alpha: float, ssaa: int, num_classes: int,
                id_color_scale: float):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    head.eval()
    import imageio.v2 as imageio
    def id2color(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x * id_color_scale)

    for idx in indices:
        item = ds[idx]
        viewmat = item["viewmat"].to(device).float()
        K = item["K"].to(device).float()
        img = item["image"]; img_t = torch.from_numpy(img).to(device).float() if isinstance(img, np.ndarray) else img.to(device).float()
        if "height" in item and "width" in item: H,W = int(item["height"]), int(item["width"])
        else: H,W = int(img_t.shape[0] if img_t.dim()==3 and img_t.shape[-1] in (3,4) else img_t.shape[1]), int(img_t.shape[1] if img_t.dim()==3 and img_t.shape[-1] in (3,4) else img_t.shape[2])

        feat, _, _ = render_feat_depth_alpha(means, quats, scales, opacities, id2color(id_codes), viewmat, K, W, H, ssaa=ssaa, camera_model=camera_model)
        logits = head(feat)[..., :num_classes]
        pred = logits.argmax(dim=-1)
        color = colorize_labels(pred)
        if overlay:
            rgb = to_hwc_uint8(img)
            if rgb.shape[:2] != color.shape[:2]:
                t = torch.from_numpy(color).permute(2,0,1)[None].float()
                t = F.interpolate(t, size=(rgb.shape[0], rgb.shape[1]), mode="nearest")
                color = t.squeeze(0).permute(1,2,0).byte().numpy()
            blend = (alpha*color + (1-alpha)*rgb).astype(np.uint8)
            out_img = blend
        else:
            out_img = color
        name = Path(item.get("path", f"{idx:06d}")).stem
        out_path = out / f"{name}.png"
        imageio.imwrite(out_path.as_posix(), out_img)
        print(f"[eval] wrote: {out_path.as_posix()}")


# -----------------------------
# main
# -----------------------------
def main():
    ap = torch.optim.argparse.ArgumentParser() if hasattr(torch.optim, "argparse") else __import__("argparse").ArgumentParser()
    A = ap.add_argument

    # data/ckpt
    A("--data_dir", type=str, required=True)
    A("--data_factor", type=int, default=4)
    A("--mask_dir", type=str, required=True)
    A("--images_dir", type=str, default=None)
    A("--ckpt", type=str, required=True)
    A("--result_dir", type=str, default="results/grouping_run")

    # model/optim
    A("--id_dim", type=int, default=32)
    A("--max_classes", type=int, default=256)
    A("--fixed_num_classes", type=int, default=None)
    A("--batch_size", type=int, default=1)
    A("--epochs", type=int, default=3)
    A("--num_workers", type=int, default=0)
    A("--camera_model", type=str, default="pinhole")

    A("--lr_id", type=float, default=1e-3)
    A("--lr_head", type=float, default=1e-3)
    A("--warmup_steps", type=int, default=800)
    A("--head_warmup_steps", type=int, default=200)
    A("--max_steps", type=int, default=12000)
    A("--clip_grad", type=float, default=1.0)

    # visibility / masking
    A("--ignore_index", type=int, default=255)
    A("--visible_alpha_thresh", type=float, default=0.05)
    A("--min_fg_ratio", type=float, default=0.01)
    A("--min_instance_area", type=int, default=64)

    # tv/edge + entropy/balance + diversity
    A("--tv_weight", type=float, default=5e-4)
    A("--edge_weight", type=float, default=1.0)
    A("--ent_weight", type=float, default=5e-4)
    A("--bg_ent_weight", type=float, default=5e-3)
    A("--cls_bal_weight", type=float, default=1e-2)
    A("--feat_div_weight", type=float, default=1e-2)

    # alt freeze
    A("--alt_cycle", type=int, default=800)
    A("--alt_freeze_head_steps", type=int, default=500)

    # id→color mapping
    A("--id_color_scale", type=float, default=2.5)
    A("--norm_id", action="store_true")

    # photometric/geo priors (optional)
    A("--geo_enable", action="store_true")
    A("--prior_depth_dir", type=str, default=None)
    A("--prior_normal_dir", type=str, default=None)
    A("--lambda_d", type=float, default=0.0)
    A("--lambda_dn", type=float, default=0.0)
    A("--lambda_pho", type=float, default=0.0)
    A("--lambda_geo", type=float, default=0.0)
    A("--pho_ssim_lambda", type=float, default=0.85)
    A("--pho_pair_stride", type=int, default=1)

    # 3D class consistency (loss_cls_3d style)
    A("--reg3d_interval", type=int, default=2)
    A("--reg3d_k", type=int, default=5)
    A("--reg3d_lambda_val", type=float, default=2.0)
    A("--reg3d_max_points", type=int, default=300000)
    A("--reg3d_sample_size", type=int, default=1000)

    # eval
    A("--eval_mode", type=str, default="logits", choices=["none","logits"])
    A("--eval_indices", type=str, default="0,50,100,150")
    A("--eval_every", type=int, default=1)
    A("--eval_overlay", action="store_true")
    A("--overlay_alpha", type=float, default=0.45)
    A("--ssaa", type=int, default=2)

    # CE normalization (match repo's scale)
    A("--ce_div_by_logC", action="store_true")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.result_dir, exist_ok=True)

    # dataset / loader
    ds = GroupingDataset(data_dir=args.data_dir, factor=args.data_factor, split="train",
                         patch_size=None, mask_dir=args.mask_dir, images_dir=args.images_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # load gsplat ckpt
    base = load_ckpt(args.ckpt, device=device)
    means, quats, scales = [base[k].to(device) for k in ("means","quats","scales")]
    opacities = base["opacities"].to(device)
    N = means.shape[0]

    # identity encodings per Gaussian
    id_codes = nn.Parameter(F.normalize(torch.randn(N, args.id_dim, device=device), dim=-1))
    head = SegHead(args.id_dim, args.max_classes).to(device)
    optim = torch.optim.AdamW(
        [{"params":[id_codes], "lr":args.lr_id}, {"params":head.parameters(), "lr":args.lr_head}],
        weight_decay=1e-4
    )
    def lr_lambda(step):
        if step < args.warmup_steps: return float(step)/max(1,args.warmup_steps)
        t = (step-args.warmup_steps)/max(1,args.max_steps-args.warmup_steps)
        return 0.5*(1.0+math.cos(math.pi*min(1.0,t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # determine global classes
    def scan_max_label(mask_dir:str)->int:
        import imageio.v2 as imageio
        files = sorted(glob.glob(os.path.join(mask_dir,"*.png")) + glob.glob(os.path.join(mask_dir,"*.npy")))
        mx=0
        for f in files[:1000]:
            m = np.load(f) if f.endswith(".npy") else imageio.imread(f)
            if m.ndim == 3: m=m[...,0]
            mx = max(mx, int(np.max(m)))
        return mx+1
    if args.fixed_num_classes is not None:
        GLOBAL_C = min(args.fixed_num_classes, args.max_classes)
    else:
        GLOBAL_C = min(scan_max_label(args.mask_dir), args.max_classes)
    print(f"[class] GLOBAL_NUM_CLS={GLOBAL_C}")

    def id2color(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x * args.id_color_scale)

    # eval index parse
    def parse_indices(s: str)->List[int]:
        if not s or s in ("all","*"): return list(range(len(ds)))
        out=[]
        for tok in s.split(","):
            tok=tok.strip()
            if not tok: continue
            if "-" in tok:
                a,b = map(int, tok.split("-"))
                out.extend(list(range(min(a,b), max(a,b)+1)))
            else:
                out.append(int(tok))
        return out

    global_step=0
    it_per_epoch=len(dl)

    for ep in range(args.epochs):
        for it, batch in enumerate(dl):
            raw_step = ep*it_per_epoch + it

            # warmup/freeze id
            warm_head_only = (raw_step < args.head_warmup_steps)
            id_codes.requires_grad_(not warm_head_only)

            # alternate: freeze head
            in_alt = (raw_step >= args.head_warmup_steps) and ((raw_step-args.head_warmup_steps) % args.alt_cycle < args.alt_freeze_head_steps)
            for p in head.parameters(): p.requires_grad_(not in_alt)

            # fetch batch 0 (bs=1)
            img = batch["image"][0]
            img = torch.from_numpy(img) if isinstance(img, np.ndarray) else img
            img = img.to(device).float()
            viewmat = batch["viewmat"][0].to(device).float()
            K = batch["K"][0].to(device).float()

            gt = batch["mask"][0]
            gt = torch.from_numpy(gt) if not isinstance(gt, torch.Tensor) else gt
            gt = gt.to(device).long()
            H, W = int(gt.shape[0]), int(gt.shape[1])

            # frame filter
            valid_pix0 = (gt != args.ignore_index)
            fg_ratio = valid_pix0.float().mean().item()
            if fg_ratio < args.min_fg_ratio: continue

            # prune tiny instances
            if args.min_instance_area > 0:
                fg = (gt != args.ignore_index) & (gt > 0)
                if fg.any():
                    vals, cnts = torch.unique(gt[fg], return_counts=True)
                    small = vals[cnts < args.min_instance_area]
                    if len(small) > 0:
                        gt = gt.clone()
                        for s in small.tolist(): gt[gt==int(s)] = args.ignore_index

            # clamp labels into [0,C)
            overflow = (gt >= GLOBAL_C) | (gt < 0)
            if overflow.any():
                gt = gt.clone(); gt[overflow] = args.ignore_index
            valid_pix = (gt != args.ignore_index)

            # render features/depth/alpha
            feat, depth, alpha = render_feat_depth_alpha(
                means, quats, scales, opacities, id2color(id_codes), viewmat, K, W, H, ssaa=args.ssaa, camera_model=args.camera_model
            )
            visible = (alpha > args.visible_alpha_thresh)
            valid_pix = valid_pix & visible

            # head
            logits = head(feat)[..., :GLOBAL_C]  # [H,W,C]

            # edges (Sobel) for TV weighting
            with torch.no_grad():
                if img.dim()==3 and img.shape[-1] in (3,4):
                    gray = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).unsqueeze(0).unsqueeze(0)
                else:
                    gray = (gt.float().unsqueeze(0).unsqueeze(0) > 0).float()
                kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=device, dtype=gray.dtype).view(1,1,3,3)
                ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=device, dtype=gray.dtype).view(1,1,3,3)
                gx = F.conv2d(gray, kx, padding=1); gy = F.conv2d(gray, ky, padding=1)
                edges = torch.sqrt(gx**2+gy**2).squeeze()
                edges = (edges - edges.min())/(edges.max()-edges.min()+1e-8)
                wmap = (1.0 + args.edge_weight * edges) * valid_pix.float()

            # ----- losses -----
            C = logits.shape[-1]
            gt = gt.clone(); gt[(gt < 0) | (gt >= C)] = args.ignore_index

            # CE (Focal + class-balance) + optional /log(C)
            logits_flat = logits.view(-1, C)
            gt_flat = gt.view(-1)
            w_flat = wmap.view(-1)
            valid = (gt_flat != args.ignore_index)
            if valid.any():
                cls, cnt = torch.unique(gt_flat[valid], return_counts=True)
                freq = torch.zeros(C, device=device); freq[cls.long()] = cnt.float()
                cls_w = torch.ones(C, device=device); m = freq > 0
                cls_w[m] = freq[m].rsqrt(); cls_w = cls_w/(cls_w.mean()+1e-8)

                lv = logits_flat[valid]; tv = gt_flat[valid]; wf = w_flat[valid]; cw = cls_w[tv]
                logp = F.log_softmax(lv, dim=-1); p = logp.exp()
                pt = p.gather(1, tv[:,None]).squeeze(1)
                ce = F.nll_loss(logp, tv, reduction="none")
                focal = (1-pt).pow(2.0) * ce
                loss_ce = (focal * cw * wf).sum() / (wf.sum()+1e-8)
                if args.ce_div_by_logC: loss_ce = loss_ce / math.log(C+1e-8)
            else:
                loss_ce = torch.tensor(0.0, device=device)

            prob = F.softmax(logits, dim=-1)
            # TV on prob (visible frontier)
            m = valid_pix.float()
            mx = m[1:,:]*m[:-1,:]; my = m[:,1:]*m[:,:-1]
            dx = (prob[1:,:,:]-prob[:-1,:,:]).abs().sum(-1); dy = (prob[:,1:,:]-prob[:,:-1,:]).abs().sum(-1)
            wx = (edges[1:,:]*edges[:-1,:]); wy = (edges[:,1:]*edges[:,:-1])
            loss_tv = args.tv_weight * ((dx*mx*wx).sum()/(mx.sum()+1e-8) + (dy*my*wy).sum()/(my.sum()+1e-8))

            # entropies
            loss_ent = torch.tensor(0.0, device=device)
            if args.ent_weight>0 and valid_pix.any():
                pv = prob[valid_pix]+1e-8
                ent = -(pv*pv.log()).sum(-1).mean()
                loss_ent = args.ent_weight * ent

            loss_bgent = torch.tensor(0.0, device=device)
            bg = (gt == args.ignore_index)
            if args.bg_ent_weight>0 and bg.any():
                pb = prob[bg]+1e-8
                ent_bg = -(pb*pb.log()).sum(-1).mean()
                loss_bgent = - args.bg_ent_weight * ent_bg

            loss_clsbal = torch.tensor(0.0, device=device)
            if args.cls_bal_weight>0 and valid_pix.any():
                p_mean = prob[valid_pix].mean(dim=(0,1))
                ent_m = -(p_mean*(p_mean+1e-8).log()).sum()
                loss_clsbal = - args.cls_bal_weight * ent_m

            # feature diversity
            if args.feat_div_weight>0 and valid_pix.any():
                z = feat[valid_pix].reshape(-1, feat.shape[-1])
                var = z.var(dim=0, unbiased=False).mean()
                loss_featdiv = - args.feat_div_weight * var
            else:
                loss_featdiv = torch.tensor(0.0, device=device)

            # tiny L2 on id
            loss_reg = (id_codes**2).mean() * 1e-4

            # photometric / depth / normal / geo (optional; mild)
            loss_d = torch.tensor(0.0, device=device)
            loss_dn = torch.tensor(0.0, device=device)
            loss_pho = torch.tensor(0.0, device=device)
            loss_geo = torch.tensor(0.0, device=device)
            if args.geo_enable and (args.lambda_d+args.lambda_dn+args.lambda_pho+args.lambda_geo)>0:
                # (c) photometric with a neighbour frame
                if args.lambda_pho>0:
                    ref_idx = ((batch.get("index",[it])[0] if isinstance(batch.get("index",[it]), list) else it) + int(args.pho_pair_stride)) % len(ds)
                    item_r = ds[ref_idx]
                    viewmat_r = item_r["viewmat"].to(device).float()
                    K_r = item_r["K"].to(device).float()
                    rgb_t = torch.from_numpy(to_hwc_uint8(img).astype(np.float32)/255.0).to(device)
                    rgb_r = torch.from_numpy(to_hwc_uint8(item_r["image"]).astype(np.float32)/255.0).to(device)
                    # warp ref->tgt (simple)
                    Hh, Ww = H, W
                    xg, yg = meshgrid_xy(Hh, Ww, device)
                    ones = torch.ones_like(xg)
                    pix = torch.stack([xg,yg,ones], dim=-1).view(-1,3)
                    Kt_inv = torch.inverse(K)
                    dirs = (Kt_inv @ pix.T).T
                    dirs = dirs / (dirs[:,2:3].clamp(min=1e-6))
                    Twc = torch.inverse(viewmat)
                    Rwc, cw = Twc[:3,:3], Twc[:3,3]
                    Xw = cw[None,:] + (Rwc @ dirs.T).T * depth.view(-1,1)
                    Rcw_r, tcw_r = viewmat_r[:3,:3], viewmat_r[:3,3]
                    Xr = (Rcw_r @ Xw.T).T + tcw_r[None,:]
                    zr = Xr[:,2].clamp(min=1e-6)
                    proj = (K_r @ (Xr/zr[:,None]).T).T
                    u, v = proj[:,0].view(Hh,Ww), proj[:,1].view(Hh,Ww)
                    un = (u/(Ww-1)*2-1).clamp(-1,1); vn = (v/(Hh-1)*2-1).clamp(-1,1)
                    grid = torch.stack([un,vn], dim=-1).unsqueeze(0)
                    Cr = F.grid_sample(rgb_r.permute(2,0,1)[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0].permute(1,2,0)
                    mask = (u>=0)&(u<=Ww-1)&(v>=0)&(v<=Hh-1)
                    lamb = float(args.pho_ssim_lambda)
                    ssim_v = _ssim(rgb_t, Cr)
                    l1 = torch.abs(rgb_t - Cr)[mask].mean() if mask.any() else torch.tensor(0.0, device=device)
                    loss_pho = (1.0 - ssim_v) * lamb / 2.0 + (1.0 - lamb) * l1

                # (d) simple geo reprojection consistency (tgt->ref)
                if args.lambda_geo>0:
                    # reuse u,v,zr from above if available else compute similarly; for brevity we reuse
                    loss_geo = loss_pho*0.0  # keep placeholder if you disable pho; set your own geo term here

                # (a) (b) priors: left as optional hooks (depth/normal)
                # keep zeros if not provided/desired

            # 3D class consistency (loss_cls_3d style)
            loss_3d = torch.tensor(0.0, device=device)
            if (raw_step % max(1,args.reg3d_interval) == 0) and args.reg3d_lambda_val>0:
                loss_3d = knn_class_consistency(
                    means, id_codes, head,
                    k=args.reg3d_k, lam=args.reg3d_lambda_val,
                    max_points=args.reg3d_max_points, sample_size=args.reg3d_sample_size
                )

            loss = (loss_ce + loss_tv + loss_ent + loss_bgent + loss_clsbal + loss_featdiv + loss_reg
                    + args.lambda_d*loss_d + args.lambda_dn*loss_dn + args.lambda_pho*loss_pho + args.lambda_geo*loss_geo
                    + loss_3d)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            # quick grad prints (watch id grads)
            id_gn = 0.0 if (id_codes.grad is None) else float(id_codes.grad.norm())
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + [id_codes], max_norm=args.clip_grad)
            optim.step()
            if args.norm_id and id_codes.requires_grad:
                with torch.no_grad(): id_codes.data = F.normalize(id_codes.data, dim=-1)
            scheduler.step()

            if global_step % 50 == 0:
                # id update monitor
                if not hasattr(main, "_prev"): main._prev = id_codes.detach().clone()
                delta = (id_codes - main._prev).abs().mean().item(); main._prev = id_codes.detach().clone()
                print(f"[ep {ep} it {it}] loss={loss.item():.4f} (ce={loss_ce.item():.4f}, tv={float(loss_tv):.1e}, ent={float(loss_ent):.4f}, bgent={float(loss_bgent):.4f}, clsbal={float(loss_clsbal):.4f}, div={float(loss_featdiv):.4f}, 3d={float(loss_3d):.4f}, pho={float(loss_pho):.4f}) id∇={id_gn:.2e} Δ={delta:.2e}")

            global_step += 1

        # save ckpt (compatible-ish)
        ckpt_path = os.path.join(args.result_dir, f"grouping_ep{ep:02d}.pt")
        torch.save(
            {
                "splats": {
                    "means": means.detach().cpu(),
                    "quats": quats.detach().cpu(),
                    "scales": torch.log(scales.detach().cpu()),
                    "opacities": torch.logit(opacities.detach().cpu().clamp(1e-6, 1-1e-6)),
                    "sh0": base.get("sh0", None),
                    "shN": base.get("shN", None),
                },
                "grouping": {
                    "id_codes": id_codes.detach().cpu(),
                    "id_dim": args.id_dim,
                    "seg_head": head.state_dict(),
                    "num_classes": args.max_classes,
                    "global_num_classes": GLOBAL_C,
                },
                "meta": {"source_ckpt": args.ckpt},
            },
            ckpt_path
        )
        print(f"[save] {ckpt_path}")

        # eval
        if args.eval_mode == "logits" and ((ep+1) % args.eval_every == 0):
            idxs = parse_indices(args.eval_indices)
            eval_logits(ds, means, quats, scales, opacities, id_codes, head, device,
                        out_dir=os.path.join(args.result_dir, f"eval_ep{ep:02d}"),
                        camera_model=args.camera_model, indices=idxs, overlay=args.eval_overlay,
                        alpha=args.overlay_alpha, ssaa=args.ssaa, num_classes=GLOBAL_C, id_color_scale=args.id_color_scale)

if __name__ == "__main__":
    main()
