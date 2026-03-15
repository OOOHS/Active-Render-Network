import torch
import torchvision.utils as vutils


@torch.no_grad()
def denormalize(x: torch.Tensor, mean=0.5, std=0.5):
    """
    把 [-1,1] 或 ImageNet 规范化的张量，反归一化到 [0,1] 区间。
    x: [B,C,H,W]
    mean, std: float 或 list
    """
    if isinstance(mean, (float, int)):
        mean = [mean] * x.size(1)
    if isinstance(std, (float, int)):
        std = [std] * x.size(1)

    mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, -1, 1, 1)

    x = x * std + mean  # 逆 Normalize
    return torch.clamp(x, 0.0, 1.0)


@torch.no_grad()
def make_grid_image(x: torch.Tensor, nrow=4, padding=2, normalize=False, **kwargs):
    """
    返回一个可保存的 grid 图像（Tensor）。
    """
    grid = vutils.make_grid(x, nrow=nrow, padding=padding, normalize=normalize, **kwargs)
    return grid
