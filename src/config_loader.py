"""
config_loader.py — Loads config/config.yaml and exposes cfg and paths objects.
"""

import os
import yaml
from pathlib import Path
from types import SimpleNamespace


_PROJECT_ROOT = Path(__file__).parent.parent


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def _resolve_paths(raw_paths: dict, root: Path) -> SimpleNamespace:
    """Resolve all path strings relative to project root."""
    ns = SimpleNamespace()
    for k, v in raw_paths.items():
        setattr(ns, k, root / v)
    return ns


_config_path = _PROJECT_ROOT / "config" / "config.yaml"
_raw = _load_yaml(_config_path)

cfg = _to_namespace(_raw)
paths = _resolve_paths(_raw["paths"], _PROJECT_ROOT)


def get_db_path() -> Path:
    return paths.db


def get_project_root() -> Path:
    return _PROJECT_ROOT
