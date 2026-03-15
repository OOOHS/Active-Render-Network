import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .datasets import TargetImageDataset


class PainterDataModule(pl.LightningDataModule):
    """
    基于 cfg.data 构建 train/val/test 的 DataLoader。
    支持两种目录结构：
      datasets/{name}/train|val|test/...
      datasets/{name}/*.(jpg|png)  （无 split 时退化为 train-only）
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.D = cfg.data  # shorthand，避免 data = ... 产生命名冲突

        # dataclass 方式访问
        self.batch_size  = self.D.batch_size
        self.num_workers = self.D.num_workers
        self.pin_memory  = self.D.pin_memory

        # 这三个 dataset 对象在 setup() 中创建
        self.train_set = None
        self.val_set   = None
        self.test_set  = None

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------
    def setup(self, stage=None):
        D = self.D

        datasets_root = D.datasets_root
        dataset_name  = D.dataset_name
        img_size      = D.img_size

        mean = D.normalize_mean
        std  = D.normalize_std

        # --------------------- train ---------------------
        self.train_set = TargetImageDataset(
            datasets_root=datasets_root,
            dataset_name=dataset_name,
            split="train",   # 若不存在 train/ 子目录，会 fallback 到无 split 模式
            img_size=img_size,
            augment=D.train_augment,
            normalize_mean=mean,
            normalize_std=std,
            cache_paths=D.cache_paths,
        )

        # --------------------- val ---------------------
        try:
            self.val_set = TargetImageDataset(
                datasets_root=datasets_root,
                dataset_name=dataset_name,
                split="val",
                img_size=img_size,
                augment=False,
                normalize_mean=mean,
                normalize_std=std,
                cache_paths=False,
            )
        except FileNotFoundError:
            self.val_set = None

        # --------------------- test ---------------------
        try:
            self.test_set = TargetImageDataset(
                datasets_root=datasets_root,
                dataset_name=dataset_name,
                split="test",
                img_size=img_size,
                augment=False,
                normalize_mean=mean,
                normalize_std=std,
                cache_paths=False,
            )
        except FileNotFoundError:
            self.test_set = None

    # ----------------------------------------------------
    # Dataloaders
    # ----------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_set is None:
            return None
        return DataLoader(
            self.val_set,
            batch_size=min(self.batch_size, 8),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        if self.test_set is None:
            return None
        return DataLoader(
            self.test_set,
            batch_size=min(self.batch_size, 8),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
