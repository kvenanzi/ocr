"""Dataset preparation for all tracks. Output is cached under DATA_DIR/<profile>/<track>/."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


# Bump when loaders change what they produce, so cached data gets rebuilt.
DATA_FORMAT = 1


def track_dir(profile_name: str, track: str) -> Path:
    from .. import config   # imported lazily: workers in engine venvs import ocrbench.data.base only

    return config.DATA_DIR / profile_name / track


def prepare_tracks(profile_name: str, tracks: list[str], force: bool = False) -> dict[str, Path]:
    from .. import config
    from . import finebooks, iam, olmocr_bench, synthetic

    prof = config.profile(profile_name)
    loaders = {"books": finebooks.prepare, "docs": olmocr_bench.prepare,
               "handwriting": iam.prepare, "stress": synthetic.prepare}
    manifests = {}
    for track in tracks:
        out = track_dir(profile_name, track)
        manifest, stamp = out / "manifest.jsonl", out / "prepared.json"
        want = {"pages": prof["pages"][track], "seed": prof["seed"], "format": DATA_FORMAT}
        if force or not manifest.exists() or not stamp.exists() or json.loads(stamp.read_text()) != want:
            shutil.rmtree(out, ignore_errors=True)
            print(f"Preparing {track} ({want['pages']} pages) -> {out}", flush=True)
            loaders[track](out, want["pages"], want["seed"])
            stamp.write_text(json.dumps(want))
        manifests[track] = manifest
    return manifests
