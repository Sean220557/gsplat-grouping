# Grouping（基于 gsplat 的 Gaussian Grouping 复现）使用说明 & 复现思路

## 0. 环境准备

**依赖**：Python 3.9–3.11（建议 3.10）、PyTorch（与 CUDA 匹配）、C++/CUDA 编译环境。
 **安装 gsplat 与本项目依赖：**

```bash
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat

pip install -e .
pip install segment-anything imageio opencv-python matplotlib timm
```

## 1. 下载数据集（与 gsplat 对齐）

```bash
cd examples

python datasets/download_dataset.py
```

目录示例：

```
data/360_v2/garden/
├─ images/ 
├─ images_4/ 
├─ sparse/
├─ cameras.json / transforms.json / ...
```

------

## 2. SAM 权重下载 & 预计算掩码

**下载 SAM 权重 `sam_vit_h_4b8939.pth`（Windows PowerShell 示例）：**

```powershell
$p = "grouping/models/sam/sam_vit_h_4b8939.pth"
New-Item -ItemType Directory -Force -Path (Split-Path $p) | Out-Null
(New-Object Net.WebClient).DownloadFile(
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", $p)
```

**生成实例掩码（PNG，文件名与 `images_4` 对齐）：**

```bash
python grouping/sam_precompute.py \
  --data_dir   data/360_v2/garden \
  --images_dir data/360_v2/garden/images_4 \
  --mask_dir   data/360_v2/garden/masks \
  --sam_model  vit_h \
  --sam_ckpt   grouping/models/sam/sam_vit_h_4b8939.pth \
  --max_instances 60
```

------

## 3. 掩码检测

```bash
python grouping/verify_masks.py \
  --data_dir data/360_v2/garden \
  --mask_dir data/360_v2/garden/masks \
  --factor   4
```

期望：**MISSING≈0**、**OK>0**、ALL_ZERO 尽量少。

------

## 4.训练基础高斯：运行 gsplat 的 `simple_trainer.py` 产出渲染 ckpt

```bash
python simple_trainer.py \
  --data_dir   data/360_v2/garden \
  --data_factor 4 \
  --result_dir results/garden \
  --iters 7000 \
  --save_every 1000 \
  --width 800 --height 800 
```

**产出示例：**

```
results/garden/
└─ ckpts/
   ├─ ckpt_0999_rank0.pt
   ├─ ckpt_1999_rank0.pt
   └─ ckpt_6999_rank0.pt
```

------

## 5. 制备单目先验：运行 `make_priors.py`（用于可选的几何正则）

> 若你计划启用 `--geo_enable`（GOC 4.2 Geometry Regularization），建议先生成深度/法线先验。

**目录结构（训练时会读取）：**

```
data/360_v2/garden/priors/
├─ depth/   000000.npy|png  (H×W，float；单位不必是米，训练会做 scale+shift 对齐)
└─ normal/  000000.npy|png  (H×W×3，[-1,1]；PNG 0..255 会自动映射)
```

**安装一个深度后端（任选其一）**

```bash
# 方案 A：MiDaS
pip install git+https://github.com/isl-org/MiDaS.git
# 方案 B：Depth-Anything v2
pip install git+https://github.com/isl-org/Depth-Anything.git
```

**运行（MiDaS 示例）**

```bash
python grouping/make_priors.py \
  --data_dir   data/360_v2/garden \
  --images_dir data/360_v2/garden/images_4 \
  --out_root   data/360_v2/garden/priors \
  --backend    midas \
  --device     cuda
```

**运行（Depth-Anything v2 示例）**

```bash
python grouping/make_priors.py \
  --data_dir   data/360_v2/garden \
  --images_dir data/360_v2/garden/images_4 \
  --out_root   data/360_v2/garden/priors \
  --backend    depth_anything_v2_base \
  --device     cuda
```

> 法线默认由深度+K 计算得到；若你已有法线网络，也可直接把结果写入 `priors/normal/`。

------

## 6. 训练 Grouping（稳边界 + 几何正则可选）

**标准训练（稳边界 & 硬标签评测）：**

```bash
python grouping/group_finetune.py \
  --data_dir data/360_v2/garden \
  --data_factor 4 \
  --mask_dir data/360_v2/garden/masks \
  --images_dir data/360_v2/garden/images_4 \
  --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
  --result_dir results/garden_grouping \
  --id_dim 32 \
  --lr_id 3e-3 --lr_head 1e-3 \
  --warmup_steps 500 --max_steps 20000 \
  --clip_grad 1.0 \
  --tv_weight 5e-4 --edge_weight 1.0 \
  --min_fg_ratio 0.01 \
  --ssaa 2 \
  --epochs 3 \
  --eval_mode logits \
  --eval_indices 0,10,20 \
  --eval_overlay --overlay_alpha 0.35
```

**启用几何正则（可选）：**

```bash
python grouping/group_finetune.py \
  ...（与上相同）... \
  --geo_enable \
  --prior_depth_dir  data/360_v2/garden/priors \
  --prior_normal_dir data/360_v2/garden/priors \
  --lambda_d 0.3 --lambda_dn 0.1 --lambda_pho 0.3 --lambda_geo 0.3 \
  --pho_ssim_lambda 0.85 --pho_pair_stride 1
```

> 若你本地 gsplat 版本不支持 `render_mode="RGB+ED"`，脚本会自动回退为仅特征渲染（几何项影响减弱）。

------

## 7. 可视化

**分组特征伪彩（PCA）：**

```bash
python grouping/render_groups.py \
  --ckpt results/garden_grouping/grouping_ep02.pt \
  --data_dir data/360_v2/garden \
  --data_factor 4 \
  --index 0 \
  --output results/garden_grouping/seg_preview.png
```
------

## 8. 复现思路简介

1. 先用 **simple_trainer** 学出稳定的 3D 高斯场（几何/外观）。
2. 在该 ckpt 上为每个高斯学习 **`id_codes ∈ ℝ^D`**，用 gsplat 光栅化得到 **每像素 D 维特征**。
3. 以 **SAM 掩码**为监督（ignore 背景），配合 **边缘加权 CE + TV 正则 + SSAA** 稳定收敛。
4. （可选）引入 **单目深度/法线先验 + 跨视角重投影** 的几何正则，使分组与几何更一致、边界更贴。
5. 通过 **硬标签渲染** 与 **PCA 伪彩** 进行目测评测与调参。

------

