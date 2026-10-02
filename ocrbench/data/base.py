"""The Sample record shared by every track, and manifest (JSONL) I/O."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Sample:
    id: str                 # unique within a track
    track: str              # books | docs | handwriting | stress
    image: str              # path relative to the track directory
    gt_text: str = ""       # empty for docs (scored by unit tests instead)
    meta: dict = field(default_factory=dict)


def write_manifest(track_dir: Path, samples: list[Sample]) -> Path:
    track_dir.mkdir(parents=True, exist_ok=True)
    path = track_dir / "manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: Path) -> list[Sample]:
    with open(path, encoding="utf-8") as f:
        return [Sample(**json.loads(line)) for line in f if line.strip()]


def stratified_sample(groups: dict[str, list], n: int, rng) -> list:
    """Round-robin across groups so every book/category/language is represented."""
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(groups.items())}
    out: list = []
    while len(out) < n and any(pools.values()):
        for k in list(pools):
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out
