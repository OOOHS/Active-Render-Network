import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    标准向量量化 (VQ-VAE 核心模块)
    - 最近邻查找 (nearest neighbor)
    - 直通梯度 (STE)
    - 两种损失:
        commit_loss    给 encoder / actor
        codebook_loss  给 codebook 更新
    - 返回 perplexity / 使用的 code 数量 / mean usage
    """
    def __init__(self, codebook_size: int = 1024, token_dim: int = 256, beta: float = 0.25):
        super().__init__()
        self.K = int(codebook_size)
        self.D = int(token_dim)
        self.beta = float(beta)

        # K×D 的可学习 embedding table 作为 codebook
        self.codebook = nn.Embedding(self.K, self.D)

        # 均匀初始化（与原项目一致）
        nn.init.uniform_(self.codebook.weight, -1.0 / self.K, 1.0 / self.K)

    # ------------------------------------------------------------
    # 最近邻查找（欧氏距离）
    # ------------------------------------------------------------
    @torch.no_grad()
    def _nearest_idx(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: [B, D]
        返回最近 code 的索引 idx[B]
        """
        E = self.codebook.weight           # [K, D]

        # 欧氏距离的展开式: ||a - e||^2 = a^2 + e^2 - 2 a·e
        a2 = (a ** 2).sum(dim=1, keepdim=True)    # [B,1]
        e2 = (E ** 2).sum(dim=1).unsqueeze(0)      # [1,K]
        ae = a @ E.t()                             # [B,K]

        dist = a2 + e2 - 2 * ae
        idx = dist.argmin(dim=1)                   # [B]
        return idx

    # ------------------------------------------------------------
    # 前向过程
    # ------------------------------------------------------------
    def forward(self, a: torch.Tensor):
        """
        输入:
            a: 连续向量 [B, D]
        输出:
            idx: 最近邻索引
            z:   STE 后的量化向量
            commit_loss
            codebook_loss
            stats: dict (perplexity / used_codes / usage_mean)
        """
        # 最近邻查找
        idx = self._nearest_idx(a)           # [B]
        z_q = self.codebook(idx)             # [B, D]

        # === 直通梯度 STE ===
        # 反向时梯度从 z 直接回到 a，不经过 codebook
        z = a + (z_q - a).detach()

        # === 两种损失 ===
        commit_loss   = self.beta * F.mse_loss(a, z_q.detach())
        codebook_loss =           F.mse_loss(a.detach(), z_q)

        # === 使用情况监控 ===
        with torch.no_grad():
            B = a.size(0)
            one_hot = torch.zeros(B, self.K, device=a.device)
            one_hot.scatter_(1, idx.view(-1, 1), 1)

            usage = one_hot.mean(dim=0)           # [K]
            eps = 1e-10
            perplexity = torch.exp(-(usage * (usage + eps).log()).sum())
            used_codes = (usage > 0).sum().float()

        stats = {
            "vq/perplexity": perplexity.detach(),
            "vq/used_codes": used_codes.detach(),
            "vq/usage_mean": usage.mean().detach(),
        }

        return idx, z, commit_loss, codebook_loss, stats


class IdentityVQ(nn.Module):
    """直通 VQ：不做量化，直接 z=a，返回与 VectorQuantizer 一致的接口。"""

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, a: torch.Tensor):
        z = a + (a.detach() - a.detach())
        zero = torch.zeros((), device=a.device, dtype=a.dtype)
        stats = {
            "vq/perplexity": torch.tensor(0.0, device=a.device),
            "vq/commit": zero,
            "vq/codebook": zero,
        }
        return None, z, zero, zero, stats

    @property
    def codebook(self):
        return nn.ParameterList([])
