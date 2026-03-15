import os, yaml, torch
import pytorch_lightning as pl
import torchvision.utils as vutils

from data.datamodule import PainterDataModule
from systems.painter import PainterSystem

@torch.no_grad()
def rollout(sys, batch, max_steps=1024, stop_tau=0.98, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    sys.eval()   # 已经够用，不需要 sys.freeze()

    device = next(sys.parameters()).device
    I_star = batch["img"].to(device)
    B, C, H, W = I_star.shape
    C_cur = torch.zeros_like(I_star)

    for t in range(max_steps):
        a = sys.actor(I_star, C_cur)
        idx, z, _ = sys.vq(a)
        delta = sys.renderer(C_cur, z)
        C_next = C_cur + delta

        # 保存当前步的画布（[-1,1] → [0,1]）
        img_to_save = (C_next.clamp(-1, 1) + 1) / 2.0
        grid = vutils.make_grid(img_to_save, nrow=B)
        vutils.save_image(grid, os.path.join(outdir, f"step_{t:04d}.png"))

        # 终止条件
        sim = sys.sim_fn(C_next, I_star).mean().item()
        if sim >= stop_tau:
            print(f"Stopped at step {t}, similarity={sim:.3f}")
            break
        C_cur = C_next

def main(cfg_path="configs/default.yaml", ckpt_path="last.ckpt"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pl.seed_everything(cfg.get("seed", 42))

    dm = PainterDataModule(cfg)
    sys = PainterSystem.load_from_checkpoint(ckpt_path, cfg=cfg)

    batch = next(iter(dm.train_dataloader()))
    rollout(
        sys, batch,
        max_steps=cfg["train"]["horizon"],
        stop_tau=cfg["train"]["stop_tau"]
    )

if __name__ == "__main__":
    import sys as _sys
    cfg_path = _sys.argv[1] if len(_sys.argv) > 1 else "configs/default.yaml"
    ckpt_path = _sys.argv[2] if len(_sys.argv) > 2 else "last.ckpt"
    main(cfg_path, ckpt_path)
