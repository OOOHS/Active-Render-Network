# modules/reward_discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  ResBlock（无归一化，适配 WGAN-GP）
# ============================================================
class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, down=True):
        super().__init__()
        self.down = down

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        # skip 连接
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()

        # 下采样
        self.downsample = nn.AvgPool2d(2) if down else nn.Identity()

    def forward(self, x):
        h = F.leaky_relu(x, 0.2, inplace=True)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.conv2(h)

        if self.down:
            h = self.downsample(h)

        s = self.skip(x)
        if self.down:
            s = self.downsample(s)

        return h + s


# ============================================================
#  PatchGAN Discriminator（ResNet backbone）
# ============================================================
class Discriminator(nn.Module):
    """
    PatchGAN 风格判别器：
      - head 输出 patch score map: [B,1,H',W']
      - forward 默认 reduce='mean' 返回 [B]（兼容旧代码）
      - forward_map 可取完整 patch map
    支持可选条件：cond != None 时在通道维 concat: [x, cond]
    """
    def __init__(self, in_ch=3, ch=64, reduce: str = "mean"):
        super().__init__()
        assert reduce in ["mean", "sum", "none"]
        self.in_ch = in_ch
        self.reduce = reduce

        self.stem = nn.Conv2d(in_ch, ch, 3, 1, 1)

        self.block1 = ResBlockD(ch,     ch * 2, down=True)
        self.block2 = ResBlockD(ch * 2, ch * 4, down=True)
        self.block3 = ResBlockD(ch * 4, ch * 8, down=True)
        self.block4 = ResBlockD(ch * 8, ch * 8, down=True)
        self.block5 = ResBlockD(ch * 8, ch * 8, down=False)

        # Patch score head
        self.head = nn.Conv2d(ch * 8, 1, 1, 1, 0)

    def _cat_cond(self, x, cond):
        if cond is None:
            return x
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, cond], dim=1)

    def forward_map(self, x, cond=None):
        """
        返回 patch score map: [B,1,H',W']
        """
        x = self._cat_cond(x, cond)

        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)

        s_map = self.head(F.leaky_relu(h, 0.2, inplace=True))  # [B,1,H',W']
        return s_map

    def forward(self, x, cond=None):
        """
        默认返回 [B]（reduce over patches），兼容旧训练代码。
        若需要 patch map，用 forward_map。
        """
        s_map = self.forward_map(x, cond=cond)  # [B,1,H',W']
        if self.reduce == "none":
            return s_map
        if self.reduce == "sum":
            return s_map.sum(dim=(1, 2, 3))
        return s_map.mean(dim=(1, 2, 3))


# ============================================================
#  WGAN-GP (PatchGAN compatible)
# ============================================================
def gradient_penalty(D: Discriminator, real, fake, cond=None):
    """
    real/fake: 输入到 D 的 x 部分（不包含 cond）
    cond:      条件图（不插值）
    """
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=real.device, dtype=real.dtype)
    inter = eps * real + (1 - eps) * fake
    inter.requires_grad_(True)

    # 使用 patch map 的均值作为每个样本的标量输出，符合 WGAN-GP 的 gp 定义
    d_inter_map = D.forward_map(inter, cond=cond) if cond is not None else D.forward_map(inter)
    d_inter = d_inter_map.mean(dim=(1, 2, 3))  # [B]

    grads = torch.autograd.grad(
        outputs=d_inter,
        inputs=inter,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def wgan_gp_loss(D: Discriminator, real, fake, cond=None, use_gp=True, gp_lambda=10.0):
    """
    real, fake: 只输入 x，不包含 cond
    cond 传入 D 时使用，但不做混合
    """
    d_real_map = D.forward_map(real, cond=cond) if cond is not None else D.forward_map(real)
    d_fake_map = D.forward_map(fake.detach(), cond=cond) if cond is not None else D.forward_map(fake.detach())

    # PatchGAN：对 patch 维度取均值作为批内统计
    d_real = d_real_map.mean(dim=(1, 2, 3))  # [B]
    d_fake = d_fake_map.mean(dim=(1, 2, 3))  # [B]

    loss = (d_fake.mean() - d_real.mean())

    logs = {
        "loss/D_real": d_real.mean().detach(),
        "loss/D_fake": d_fake.mean().detach(),
        "loss/D_wgan": loss.detach(),
    }

    if use_gp:
        gp = gradient_penalty(D, real, fake, cond)
        loss = loss + gp_lambda * gp
        logs["loss/D_gp"] = gp.detach()

    return loss, logs
