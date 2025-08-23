import math
from typing import Dict, Any
import torch
import torch.nn.functional as F

@torch.no_grad()
def load_ckpt(ckpt_path: str, device="cuda") -> Dict[str, Any]:
    raw = torch.load(ckpt_path, map_location=device)
    splats = raw["splats"]
    means = splats["means"]               # [N,3]
    quats = F.normalize(splats["quats"], p=2, dim=-1)  # [N,4]
    scales = torch.exp(splats["scales"])  # [N,3]
    opacities = torch.sigmoid(splats["opacities"])     # [N]
    sh0 = splats["sh0"]                   # [N,1,3]
    shN = splats["shN"]                   # [N,S-1,3]
    out = dict(
        means=means, quats=quats, scales=scales,
        opacities=opacities, sh0=sh0, shN=shN,
        meta=raw.get("meta", {}), step=raw.get("step", None)
    )
    if "grouping" in raw:
        out["grouping"] = raw["grouping"]
    return out
