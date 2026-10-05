"""Paths and YAML config loading."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
# Datasets and per-engine virtualenvs are big, so they live outside the repo on Colab.
DATA_DIR = Path(os.environ.get("OCRBENCH_DATA", REPO_ROOT / "data"))
ENVS_DIR = Path(os.environ.get("OCRBENCH_ENVS", REPO_ROOT / "envs"))

TRACKS = ("books", "docs", "handwriting", "stress")


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def profile(name: str) -> dict:
    cfg = load_yaml("profiles.yaml")
    if name not in cfg["profiles"]:
        raise KeyError(f"unknown profile {name!r}; choose from {list(cfg['profiles'])}")
    return {"name": name, "seed": cfg["seed"], **cfg["profiles"][name]}
