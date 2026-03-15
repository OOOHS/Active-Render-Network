import os
import glob
from typing import List, Optional, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T



IMG_EXTS = (
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP"
)



def list_images(root: str) -> List[str]:
    """递归列出 root 下所有图片路径。"""
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(paths)


def ensure_3ch(img: Image.Image) -> Image.Image:
    """把灰度/带Alpha统一成RGB三通道。"""
    if img.mode == "RGB":
        return img
    if img.mode in ("L", "I;16", "I;16B", "I", "F"):
        return img.convert("RGB")
    if img.mode == "RGBA":
        return img.convert("RGB")
    # 其余模式兜底
    return img.convert("RGB")


class TargetImageDataset(Dataset):
    """
    通用图片数据集：
    - 根目录形如: datasets/{DATASET}/ 或 datasets/{DATASET}/{split}/
    - 若存在 split 子目录（train/val/test），优先使用；否则使用根目录下全部图片。
    - 返回: {"img": tensor[-1,1], "path": str}
    """
    def __init__(
        self,
        datasets_root: str,   # 例如 "datasets"
        dataset_name: str,    # 例如 "YourSet"
        split: Optional[str] = None,  # None / "train" / "val" / "test"
        img_size: int = 256,
        augment: bool = False,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
        cache_paths: bool = False,
    ):
        self.datasets_root = datasets_root
        self.dataset_name = dataset_name
        self.split = split

        base = os.path.join(datasets_root, dataset_name)
        if split is not None and os.path.isdir(os.path.join(base, split)):
            data_dir = os.path.join(base, split)
        else:
            # 没有显式划分，就用整个数据集目录
            data_dir = base

        self.paths = list_images(data_dir)
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found under: {data_dir}\n"
                f"Check your structure: datasets/{dataset_name}/[train|val|test]/*.jpg|png"
            )

        # 变换：train 可选增强，val/test 仅基础预处理
        tfms: List = [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),                         # [0,1]
            T.Normalize(normalize_mean, normalize_std),  # -> [-1,1] (0.5,0.5)
        ]
        aug_tfms: List = []
        if augment:
            aug_tfms = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.3),
            ]
        self.tf = T.Compose(aug_tfms + tfms)

        # 可选：把路径缓存到内存里（大数据集没必要）
        self.cache_paths = cache_paths
        if cache_paths:
            self._cached = [p for p in self.paths]  # 仅示意

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.paths[idx]
        img = Image.open(p)
        img = ensure_3ch(img)
        x = self.tf(img)  # [3,H,W], ~[-1,1]
        return {"img": x, "path": p}
