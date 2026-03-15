from dataclasses import dataclass, fields, is_dataclass
from omegaconf import DictConfig

@dataclass
class BaseConfig:
    """
    Make dataclasses behave like OmegaConf configs.
    Convert OmegaConf DictConfig -> Python dataclass recursively.
    """

    @classmethod
    def from_dict_config(cls, cfg: DictConfig):
        kwargs = {}
        for f in fields(cls):
            val = cfg.get(f.name)

            # nested dataclass
            if is_dataclass(f.type):
                kwargs[f.name] = f.type.from_dict_config(val)
            else:
                kwargs[f.name] = val

        return cls(**kwargs)
