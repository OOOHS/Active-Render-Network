from dataclasses import make_dataclass, is_dataclass
from configs.base_config import BaseConfig

def auto_dataclass_from_dict(name, d: dict):
    fields = []
    for k, v in d.items():
        if isinstance(v, dict):
            sub = auto_dataclass_from_dict(k.capitalize(), v)
            fields.append((k, sub, None))
        else:
            fields.append((k, type(v), v))
    return make_dataclass(name, fields, bases=(BaseConfig,))
