import sys
from omegaconf import OmegaConf
from configs.config import auto_dataclass_from_dict

def load_config(path="configs/default.yaml"):
    yaml_cfg = OmegaConf.load(path)
    cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
    merged = OmegaConf.merge(yaml_cfg, cli_cfg)

    # YAML -> dict
    raw = OmegaConf.to_container(merged, resolve=True)

    # 自动生成 MainConfig dataclass
    MainConfig = auto_dataclass_from_dict("MainConfig", raw)

    # dataclass 实例化（递归）
    return MainConfig.from_dict_config(merged)
