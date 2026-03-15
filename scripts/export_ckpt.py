import torch
import yaml
from systems.painter import PainterSystem

def export_actor(cfg_path="configs/default.yaml", ckpt_path="last.ckpt", out_path="actor.pt"):
    cfg = yaml.safe_load(open(cfg_path))
    sys = PainterSystem.load_from_checkpoint(ckpt_path, cfg=cfg)

    bundle = {
        "actor": sys.actor.state_dict(),
        "vq": sys.vq.state_dict(),
        "renderer": sys.renderer.state_dict(),
        "cfg": cfg["model"],
    }
    torch.save(bundle, out_path)
    print(f"Exported to {out_path}")

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "last.ckpt"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "actor.pt"
    export_actor(cfg_path, ckpt_path, out_path)
