import os
import yaml
import copy
from .paths import resolve_path


def load_config(config_path, base_config_path=None):
    """
    加载并解析配置文件
    支持继承基础配置和路径解析
    """
    # 加载主配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 处理继承关系
    if base_config_path:
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

        # 合并配置：基础配置 < 派生配置
        merged_config = {**base_config, **config}
        config = merged_config

    # 递归解析路径变量
    config = _resolve_config_paths(config, config)

    return config


def _resolve_config_paths(config, root_config):
    """递归解析配置中的所有路径"""
    if isinstance(config, dict):
        resolved = {}
        for key, value in config.items():
            resolved[key] = _resolve_config_paths(value, root_config)
        return resolved
    elif isinstance(config, list):
        return [_resolve_config_paths(item, root_config) for item in config]
    elif isinstance(config, str) and ('$' in config or '{' in config):
        # 解析路径变量
        return resolve_path(config, root_config)
    else:
        return config
