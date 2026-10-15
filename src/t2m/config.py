from pathlib import Path
import yaml


def load_config(filename):
    p = Path(filename).resolve()
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base_config" not in cfg:
        return cfg
    base_ref = Path(cfg.pop("base_config"))
    # First relative to CWD (the documented entry point), then parent YAML directory.
    base_path = base_ref if base_ref.exists() else p.parent / base_ref
    parent = load_config(base_path)
    parent.update(cfg)
    return parent
