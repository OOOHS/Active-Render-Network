import torch
import torch.nn.functional as F


@torch.no_grad()
def mse_similarity(canvas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    简单相似度 = 1 - MSE([-1,1]域)。范围约 [0,1]。
    """
    mse = F.mse_loss(canvas, target, reduction="none").mean(dim=(1,2,3))  # [B]
    sim = 1.0 - mse
    return torch.clamp(sim, 0.0, 1.0)


@torch.no_grad()
def psnr(canvas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)。输入范围 [-1,1]。
    返回 [B] 向量。
    """
    mse = F.mse_loss(canvas, target, reduction="none").mean(dim=(1,2,3))
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10 * torch.log10(4.0 / mse)  # 因为 [-1,1] 范围 -> 峰值差 2
    return psnr


@torch.no_grad()
def ssim_placeholder(canvas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    SSIM 占位符: 这里给个假实现（只返回均值差），方便接口占位。
    真正实现需引入第三方库如 piq 或 torchmetrics。
    """
    return 1.0 - (canvas.mean(dim=(1,2,3)) - target.mean(dim=(1,2,3))).abs()
