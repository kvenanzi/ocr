"""Run a per-page function across CPU cores (for single-threaded classic engines)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def cpu_workers(params: dict) -> int:
    return int(params.get("workers") or os.cpu_count() or 1)


def pmap(fn, items: list, workers: int) -> list:
    if workers <= 1 or len(items) == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
