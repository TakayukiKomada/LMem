import json
from pathlib import Path
from typing import Optional


def load_config(config_path: Optional[str] = None) -> dict:
    """設定ファイルを読み込む"""
    if config_path is None:
        config_path = "config.json"
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """2つの設定を再帰的にマージする"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict) -> bool:
    """設定の妥当性を検証する"""
    required_keys = ["model", "data_path", "output_dir"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
    if not isinstance(config["model"], str):
        raise TypeError("model must be a string")
    return True
