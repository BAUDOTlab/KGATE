import os
from typing import Sequence

from pathlib import Path
import tomllib
from importlib.resources import open_binary

class Config:
    def __init__(self, *, config_path: os.PathLike = "", config_dict: dict  = {}):
        self._config = Config.parse(config_path, config_dict)

    @staticmethod
    def parse(config_path: os.PathLike, config_dict: dict):
        if config_path != "" and not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        with open_binary("kgate", "config_template.toml") as f:
            default_config = tomllib.load(f)