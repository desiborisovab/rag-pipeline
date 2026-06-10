# load yaml file
import yaml
from pathlib import Path

_cfg = None

def get_config(path: str = "config.yml") -> dict:
    global _cfg
    if _cfg is None:
        with open(Path(path)) as f:
            _cfg = yaml.safe_load(f)
    return _cfg
