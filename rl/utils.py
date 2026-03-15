import math
import random
from typing import Optional

import numpy as np
import torch


class OUNoise:
    """
    Ornstein–Uhlenbeck 噪声（经典 DDPG 探索噪声）
    用法:
        ou = OUNoise(action_dim=D, mu=0.0, theta=0.15, sigma=0.2)
        noise = ou.sample()  # shape [D]
    """
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.state = np.ones(self.action_dim, dtype=np.float32) * self.mu

    def reset(self):
        self.state.fill(self.mu)

    def sample(self) -> torch.Tensor:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim).astype(np.float32)
        self.state = x + dx
        return torch.from_numpy(self.state.copy())  # [D]

    def sample_like(self, a: torch.Tensor) -> torch.Tensor:
        """
        给 batch 动作加 OU 噪声：返回 shape 与 a 相同的噪声（仅最后一维匹配 action_dim）
        """
        B = a.shape[0]
        n = [self.sample() for _ in range(B)]
        return torch.stack(n, dim=0).to(a.device, dtype=a.dtype)


def set_seed(seed: int):
    """
    统一设置随机种子（Python/NumPy/PyTorch）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保守选项：确保可复现（牺牲速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def clamp_actions(a: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """
    将连续动作裁剪到指定范围。
    """
    return a.clamp_(min=min_val, max=max_val)


class LinearSchedule:
    """
    线性调度器（例如探索噪声强度、学习率等）：
        sched = LinearSchedule(start=0.3, end=0.05, duration=10000)
        value = sched(step)
    """
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, int(duration))

    def __call__(self, step: int) -> float:
        t = min(1.0, max(0.0, step / self.duration))
        return (1 - t) * self.start + t * self.end
