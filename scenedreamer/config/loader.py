from omegaconf import OmegaConf


def load_cfg(cfg_path):
    """Load a YAML config as an OmegaConf object."""
    return OmegaConf.load(cfg_path)


def merge_cfgs(cfg1, cfg2):
    """Merge two OmegaConf objects, with cfg2 overriding cfg1."""
    return OmegaConf.merge(cfg1, cfg2)
