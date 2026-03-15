import torch
import torch.nn as nn
import copy


@torch.no_grad()
def make_target(net: nn.Module) -> nn.Module:
    tgt = copy.deepcopy(net)
    for p in tgt.parameters():
        p.requires_grad_(False)
    return tgt


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    for p_t, p in zip(target.parameters(), source.parameters()):
        p_t.data.lerp_(p.data, tau)
