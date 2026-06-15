from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got: {override}")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("run", "env", "train", "eval"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    if config["env"]["frame_stack"] < 1 or config["env"]["image_size"] < 36:
        raise ValueError("frame_stack must be positive and image_size must be at least 36")
    if config["train"]["n_steps"] * config["env"]["n_envs"] % config["train"]["batch_size"]:
        raise ValueError("n_steps * n_envs must be divisible by batch_size")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

