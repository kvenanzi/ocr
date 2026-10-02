"""Engine interface. Each engine runs inside a worker subprocess in its own virtualenv,
so engine modules import their heavy dependencies lazily inside `load()`."""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass
class Prediction:
    text: str
    finish_reason: str | None = None   # "length" means it hit max_tokens
    n_tokens: int | None = None


class Engine:
    # True when predict() is much faster on a list than one item at a time (vLLM, GPU
    # batching). The worker then measures latency (batch=1) and throughput (full
    # batch) separately; otherwise one sequential pass yields both.
    native_batch = False
    batch_size = 1

    def __init__(self, spec: dict, hardware: dict):
        self.spec = spec
        self.params = spec.get("params", {})
        self.hw = hardware

    @property
    def use_gpu(self) -> bool:
        return self.hw.get("kind") == "gpu"

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, image_paths: list[str]) -> list[Prediction]:
        raise NotImplementedError

    def versions(self) -> dict:
        """Package versions worth recording next to the results."""
        return {}


def pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def create(spec: dict, hardware: dict) -> Engine:
    module, _, cls = spec["impl"].partition(":")
    mod = importlib.import_module(f"ocrbench.engines.{module}")
    return getattr(mod, cls or "ENGINE")(spec, hardware)
