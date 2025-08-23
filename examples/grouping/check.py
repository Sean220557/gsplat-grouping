# examples/grouping/check.py
import argparse, glob
from pathlib import Path
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    import imageio
from PIL import Image

import torch

import sys
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES_DIR))
# ---- 可选：仅在 --ckpt 时需要 ----
try:
    from gsplat.rendering import rasterization
    from datasets.colmap import Dataset as ColmapDataset, Parser as ColmapParser
    from grouping.load_ckpt import load_ckpt
except Exception:
    ColmapDataset = None
    ColmapParser = None
    rasterization = None
    load_ckpt = None


def pick_images_dir(root: Path, factor: int, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.is_dir():
            raise FileNotFoundError(f"--images_dir not found: {p}")
        return p
    root = Path(root)
    cands = [root / f"images_{factor}_png"] + sorted(root.glob(f"images_{factor}*")) + [root / "images"]
    cands = [p for p in cands if p.exists() and p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"No images dir under {root}. Tried images_{factor}_png, images_{factor}*, images/")
    def _score(p: Path):
        if p.name == f"images_{factor}_png": return (0, p.name)
        if p.name == f"images_{factor}":     return (1, p.name)
        if p.name.startswith(f"images_{factor}"): return (2, p.name)
        if p.name == "images":               return (3, p.name)
        return (4, p.name)
    cands.sort(key=_score)
    return cands[0]


def list_images(imgdir: Path) -> list[Path]:
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        files += glob.glob(str(imgdir / ext))
    files = [Path(f) for f in sorted(files)]
    if not files:
        raise FileNotFoundError(f"No images found in {imgdir}")
    return files


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".npy"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists(): return p
    hits = sorted(mask_dir.glob(f"{stem}*.png")) + sorted(mask_dir.glob(f"{stem}*.npy"))
    return hits[0] if hits else None


def read_mask_any(p: Path) -> np.ndarray:
    if p.suffix.lower() == ".npy":
        m = np.load(p.as_posix())
    else:
        m = imageio.imread(p.as_posix())
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.int64)


def resize_nn(arr: np.ndarray, size_hw: tuple[int,int]) -> np.ndarray:
    H, W = size_hw
    if arr.shape[:2] == (H, W): return arr
    im = Image.fromarray(arr)
    im = im.resize((W, H), resample=Image.NEAREST)
    return np.array(im)


def _get_viewmat(item: dict) -> torch.Tensor:
    # 直接的 view 矩阵
    for k in ("viewmat", "w2c", "world_view_transform"):
        if k in item:
            return item[k].float()
    # 需要 inverse 的 world 矩阵
    for k in ("camtoworld", "c2w", "camera_to_world", "world_to_camera_inv"):
        if k in item:
            return torch.inverse(item[k].float())
    raise KeyError("No view matrix key among ['viewmat','w2c','world_view_transform','camtoworld','c2w','camera_to_world'].")


def _get_K(item: dict) -> torch.Tensor:
    if "K" in item:
        return item["K"].float()
    if "intrinsics" in item:
        fx, fy, cx, cy = item["intrinsics"]
        K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=torch.float32, device=fx.device)
        return K
    # 兜底（不常用）
    for k in ("focal", "fx", "fy", "cx", "cy"):
        if k in item:
            fx = float(item.get("fx", item.get("focal", 1.0)))
            fy = float(item.get("fy", fx))
            cx = float(item.get("cx", 0.5))
            cy = float(item.get("cy", 0.5))
            return torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=torch.float32)
    raise KeyError("No intrinsics 'K' found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--images_dir", type=str, default=None)
    # 可选：alpha 覆盖 + 双向 overlay
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--alpha_th", type=float, default=0.05)
    ap.add_argument("--alpha_out", type=str, default=None,
                    help="输出文件前缀路径（不要带 _w2c/_c2w_inv），会自动创建父目录")
    ap.add_argument("--camera_model", type=str, default="pinhole")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    mask_dir = Path(args.mask_dir)
    img_dir = pick_images_dir(data_dir, args.factor, args.images_dir)
    imgs = list_images(img_dir)

    if args.idx < 0 or args.idx >= len(imgs):
        raise IndexError(f"--idx {args.idx} out of range [0, {len(imgs)-1}]")

    img_path = imgs[args.idx]
    stem = img_path.stem

    print(f"[info] images_dir = {img_dir}")
    print(f"[info] picked image[{args.idx}] = {img_path.name} (stem='{stem}')")
    print(f"[info] searching mask under {mask_dir}")

    # ---- 掩码检查 ----
    mp = find_mask(mask_dir, stem)
    if mp is None:
        print(f"[ERROR] mask missing for stem '{stem}'. Tried: {mask_dir}/{stem}.png|.npy and fuzzy '{stem}*'")
        ex = sorted(mask_dir.glob('*.png'))[:5] + sorted(mask_dir.glob('*.npy'))[:5]
        if ex: print("[hint] examples in mask_dir:", ", ".join(p.name for p in ex))
        return

    mask = read_mask_any(mp)
    # 取原图尺寸
    try:
        from PIL import Image as PILImage
        with PILImage.open(img_path.as_posix()) as im:
            W, H = im.size
    except Exception:
        arr = imageio.imread(img_path.as_posix())
        H, W = (arr.shape[0], arr.shape[1]) if arr.ndim >= 2 else (mask.shape[0], mask.shape[1])
    if mask.shape[:2] != (H, W):
        mask = resize_nn(mask, (H, W))
    ratio = float((mask != 0).sum()) / float(H * W)
    print(f"[OK] mask file = {mp.name}")
    print(f"[OK] mask>0 ratio = {ratio:.4f} (H={H}, W={W})")

    # ---- 可选：alpha 覆盖 + 双向 overlay（按同名帧匹配）----
    if args.ckpt is not None:
        if any(x is None for x in (ColmapDataset, ColmapParser, rasterization, load_ckpt)):
            print("[warn] alpha 检查所需依赖未导入，跳过。")
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        parser = ColmapParser(data_dir=args.data_dir, factor=args.factor)
        ds = ColmapDataset(parser, split="train")

        # 1) 用 stem 在 ds 里找同名帧
        target_stem = stem
        match_idx = None
        for i in range(len(ds)):
            p = ds[i].get("path", None)
            if p is not None and Path(p).stem == target_stem:
                match_idx = i
                break
        if match_idx is None:
            print(f"[warn] 在 ColmapDataset 里找不到同名帧 '{target_stem}'，改用 --idx={args.idx} 的样本。")
            match_idx = args.idx
        item = ds[match_idx]

        # 2) 相机矩阵（容错多键名）
        try:
            viewmat = _get_viewmat(item).to(device).float()
            K = _get_K(item).to(device).float()
        except KeyError as e:
            print(f"[warn] {e}  → alpha 检查跳过（不影响掩码检查）。")
            return

        if "height" in item and "width" in item:
            H2, W2 = int(item["height"]), int(item["width"])
        else:
            H2, W2 = H, W

        # 3) 载 ckpt
        ck = load_ckpt(args.ckpt, device=device)
        means, quats, scales, opacities = ck["means"], ck["quats"], ck["scales"], ck["opacities"]

        # 4) 渲 alpha 的小函数
        def render_alpha(vmat):
            dummy = torch.zeros((means.shape[0], 1), device=device, dtype=means.dtype)
            _, alphas, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities, colors=dummy,
                viewmats=vmat[None], Ks=K[None], width=W2, height=H2,
                sh_degree=None, packed=True, rasterize_mode="classic", camera_model=args.camera_model
            )
            return alphas[0, :, :, 0].clamp(0, 1)

        # 5) 读原图（用于 overlay）
        rgb = imageio.imread(img_path.as_posix())
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        # 6) 分别渲 w2c / c2w_inv
        alpha_w2c = render_alpha(viewmat)
        alpha_c2w_inv = render_alpha(torch.inverse(viewmat))

        def save_pair(a, tag):
            a_g = a.pow(0.7).detach().cpu().numpy()  # gamma=0.7 更易看
            a8 = (a_g * 255).astype(np.uint8)
            # 伪彩
            try:
                import cv2
                heat = cv2.applyColorMap(a8, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
            except Exception:
                heat = np.stack([a8]*3, axis=-1)
            ov = (0.5 * rgb + 0.5 * heat).astype(np.uint8)
            if args.alpha_out is None:
                base = Path(f"alpha_{target_stem}.png")
            else:
                base = Path(args.alpha_out)
            base.parent.mkdir(parents=True, exist_ok=True)
            g_path = base.with_name(base.stem + f"_{tag}.png")
            ov_path = base.with_name(base.stem + f"_{tag}_overlay.png")
            imageio.imwrite(g_path.as_posix(), a8)
            imageio.imwrite(ov_path.as_posix(), ov)
            print("wrote:", g_path.as_posix())
            print("wrote:", ov_path.as_posix())

        # 输出两套图
        save_pair(alpha_w2c, "w2c")
        save_pair(alpha_c2w_inv, "c2w_inv")

        # 覆盖率统计（阈值同一套）
        th = float(args.alpha_th)
        r1 = (alpha_w2c > th).float().mean().item()
        r2 = (alpha_c2w_inv > th).float().mean().item()
        print(f"[OK] alpha>{th}  w2c: {r1:.4f}   c2w_inv: {r2:.4f}  | matched stem: {target_stem} (idx={match_idx})")


if __name__ == "__main__":
    main()
