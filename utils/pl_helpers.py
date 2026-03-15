import pytorch_lightning as pl
import torch
import numpy as np
import random
import wandb

def seed_everything(seed: int = 42):
    """
    同时设置 Python/NumPy/PyTorch 随机种子。
    """
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_images(logger, tag: str, images: torch.Tensor, step: int, nrow=4):
    """
    将图像张量写入 TensorBoard / WandB。
    - logger: pl.Trainer.logger.experiment
    - images: [B,C,H,W], 已归一化到 [0,1]
    """
    if hasattr(logger, "add_images"):  # TensorBoard
        logger.add_images(tag, images, step, dataformats="NCHW")
    elif hasattr(logger, "log"):  # WandB
        import torchvision
        grid = torchvision.utils.make_grid(images, nrow=nrow)
        logger.log({tag: [wandb.Image(grid)]}, step=step)


class EMA:
    """
    Exponential Moving Average 用于模型权重平滑。
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
