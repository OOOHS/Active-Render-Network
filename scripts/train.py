import os
import torch
import pytorch_lightning as pl

from data.datamodule import PainterDataModule
from systems.painter import PainterSystem

from configs.load_config import load_config
from pytorch_lightning.callbacks import ModelCheckpoint

from swanlab.integration.pytorch_lightning import SwanLabLogger

os.environ["SWANLAB_MODE"] = "offline"

def main():
    # 1) 统一从 load_config() 加载完整 cfg
    cfg = load_config()

    # 2) seed
    pl.seed_everything(cfg.seed)

    # 3) 构建 DataModule, System（均收到 dataclass cfg）
    dm = PainterDataModule(cfg)
    sys = PainterSystem(cfg)

    # 4) Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename="{epoch}-{step}",
        every_n_train_steps=cfg.train.save_interval,
        save_last=True,
        save_top_k=-1,
        save_weights_only=False,

    )

    # 5) Trainer
    swan_logger = SwanLabLogger(project="ARN", name="0k-vit-shaping_fix-sacling-adamw", config=cfg,)
    trainer = pl.Trainer(
        max_steps=cfg.train.max_steps,
        log_every_n_steps=cfg.train.log_interval,
        val_check_interval=cfg.train.val_interval,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[2, 3],
        precision=16,
        default_root_dir="outputs",
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        logger=swan_logger,
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_true", # 参数定义但不更新
    )


    # 6) 续训逻辑
    # ckpt_path = "/home/liushudong/ARN/outputs/checkpoints/epoch=55-step=590000-v1.ckpt"
    ckpt_path = ""

    if os.path.exists(ckpt_path):
        print(f"[INFO] Resuming training from checkpoint: {ckpt_path}")
        trainer.fit(sys, datamodule=dm, ckpt_path=ckpt_path)
    else:
        print("[INFO] No checkpoint found. Starting training from scratch.")
        trainer.fit(sys, datamodule=dm)


if __name__ == "__main__":
    main()
