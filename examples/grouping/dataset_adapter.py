from pathlib import Path
from typing import Dict, Any, Optional, List
import torch, numpy as np, imageio.v2 as imageio
import glob, sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.colmap import Dataset as ColmapDataset, Parser as ColmapParser


class GroupingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        factor: int = 4,
        split: str = "train",
        patch_size: Optional[int] = None,
        mask_dir: Optional[str] = None,
        images_dir: Optional[str] = None,
    ):
        parser = ColmapParser(data_dir=data_dir, factor=factor)
        self.base = ColmapDataset(parser, split=split, patch_size=patch_size)

        if images_dir:
            img_dir = Path(images_dir)
        else:
            root = Path(data_dir)
            cand = sorted(
                [p for p in root.glob(f"images_{factor}") if p.is_dir()],
                key=lambda p: p.name,
            )
            if cand:
                img_dir = cand[-1]
            else:
                cand = sorted(
                    [p for p in root.glob("images_*") if p.is_dir()],
                    key=lambda p: p.name,
                )
                img_dir = cand[-1] if cand else root / "images"

        files = sorted(glob.glob(str(img_dir / "*.jpg"))) + sorted(
            glob.glob(str(img_dir / "*.png"))
        )
        if len(files) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.index_paths: List[Path] = [Path(f) for f in files]

        self.mask_dir = Path(mask_dir) if mask_dir else None
        self._mask_map: Dict[str, Path] = {}
        if self.mask_dir and self.mask_dir.exists():
            for p in self.mask_dir.glob("*.png"):
                self._mask_map[p.stem] = p

    def __len__(self):
        return min(len(self.base), len(self.index_paths))

    def _get_hw_from_image(self, img):
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[-1] in (3, 4):
                return int(img.shape[0]), int(img.shape[1])  # HWC
            if img.dim() == 3 and img.shape[0] in (1, 3, 4):
                return int(img.shape[1]), int(img.shape[2])  # CHW
        elif isinstance(img, np.ndarray) and img.ndim == 3:
            return int(img.shape[0]), int(img.shape[1])
        raise ValueError(
            f"Cannot infer H/W from image with shape {getattr(img,'shape',None)}"
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        path = self.index_paths[idx]  # 用统一的文件名来源
        stem = path.stem

        if "height" in item and "width" in item:
            H, W = int(item["height"]), int(item["width"])
        else:
            H, W = self._get_hw_from_image(item["image"])

        if self.mask_dir and stem in self._mask_map:
            mask_np = imageio.imread(self._mask_map[stem]).astype(np.int64)
            if mask_np.shape[:2] != (H, W):
                m = torch.from_numpy(mask_np)[None, None].float()
                m = (
                    torch.nn.functional.interpolate(m, size=(H, W), mode="nearest")
                    .squeeze()
                    .long()
                    .cpu()
                    .numpy()
                )
                mask_np = m
        else:
            mask_np = np.zeros((H, W), dtype=np.int64)

        if "viewmat" in item:
            pass
        elif "camtoworld" in item:
            item["viewmat"] = torch.inverse(torch.as_tensor(item["camtoworld"])).float()
        else:
            raise KeyError("Dataset item missing both 'viewmat' and 'camtoworld'.")

        return {**item, "mask": mask_np, "height": H, "width": W, "path": str(path)}
