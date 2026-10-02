"""Run one engine over every track and record predictions, timing and memory.

Invoked by the runner as a subprocess, using the engine family's virtualenv:

    python -m ocrbench.worker --job results/runs/<run>/<engine>/job.json

Keep imports here light (stdlib + PIL); engines import their own heavy deps.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from ocrbench.data.base import read_manifest
from ocrbench.engines.base import Prediction, create


class ResourceMonitor(threading.Thread):
    """Samples peak host RSS (this process tree) and GPU memory in use."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_rss = 0
        self.peak_vram_mib = 0
        self.vram_baseline_mib = self._vram()
        self._halt = threading.Event()
        try:
            import psutil

            self._proc = psutil.Process()
        except ImportError:
            self._proc = None

    @staticmethod
    def _vram() -> int:
        if not shutil.which("nvidia-smi"):
            return 0
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return sum(int(x) for x in out.split())
        except Exception:
            return 0

    def _rss(self) -> int:
        if self._proc is None:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        try:
            procs = [self._proc, *self._proc.children(recursive=True)]
            return sum(p.memory_info().rss for p in procs if p.is_running())
        except Exception:
            return 0

    def run(self):
        while not self._halt.is_set():
            self.peak_rss = max(self.peak_rss, self._rss())
            self.peak_vram_mib = max(self.peak_vram_mib, self._vram())
            self._halt.wait(self.interval)

    def stop(self) -> dict:
        self._halt.set()
        return {
            "peak_rss_gb": round(self.peak_rss / 1024**3, 2),
            "peak_vram_gb": round(max(self.peak_vram_mib - self.vram_baseline_mib, 0) / 1024, 2),
        }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[round(q) - 1]


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


# Errors that mean the engine process is dead (e.g. vLLM after a CUDA OOM). Every later
# call would fail instantly, so stop the engine and report an error instead of scoring
# a run of empty pages as 0% accuracy.
_FATAL = ("EngineDeadError", "OutOfMemoryError", "CUDA out of memory", "CUDA error")


def _is_fatal(e: BaseException) -> bool:
    text = f"{type(e).__name__}: {e}"
    return any(f in text for f in _FATAL)


def _predict_safely(engine, paths: list[str]):
    """Predict a chunk; if it fails, retry page by page so one bad page doesn't lose the chunk.
    Failed pages come back as empty text and are scored as errors."""
    try:
        preds = engine.predict(paths)
        if len(preds) != len(paths):
            raise RuntimeError(f"engine returned {len(preds)} predictions for {len(paths)} pages")
        return preds, [None] * len(paths)
    except Exception as e:
        if _is_fatal(e):
            raise   # the engine itself is gone; retrying page by page would only record empty pages
        if len(paths) == 1:
            traceback.print_exc()
            return [Prediction(text="")], [traceback.format_exc(limit=2)[-500:]]
    preds, errors = [], []
    for p in paths:
        pr, er = _predict_safely(engine, [p])
        preds += pr
        errors += er
    return preds, errors


def run_track(engine, samples, track_dir: Path, image_root: Path, latency_n: int, deadline: float) -> dict:
    track_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(image_root / s.image) for s in samples]
    latencies: list[float] = []

    if engine.native_batch and latency_n:
        for k, p in enumerate(paths[:latency_n], 1):
            t = time.perf_counter()
            engine.predict([p])
            latencies.append(time.perf_counter() - t)
            print(f"  [{track_dir.name}] latency page {k}/{min(latency_n, len(paths))}: {latencies[-1]:.1f}s", flush=True)
        print(f"  [{track_dir.name}] batch of {len(paths)} pages...", flush=True)

    done, partial, wall = 0, False, 0.0
    with open(track_dir / "preds.jsonl", "w", encoding="utf-8") as out:
        step = max(engine.batch_size, 1)
        for i in range(0, len(paths), step):
            if time.time() > deadline:
                partial = True
                break
            chunk = paths[i:i + step]
            t = time.perf_counter()
            preds, errors = _predict_safely(engine, chunk)
            dt = time.perf_counter() - t
            wall += dt
            if not engine.native_batch:
                latencies.append(dt / len(chunk))
            for s, pred, err in zip(samples[i:i + step], preds, errors):
                rec = {"id": s.id, "text": pred.text, "finish_reason": pred.finish_reason,
                       "n_tokens": pred.n_tokens, "seconds": dt / len(chunk)}
                if err:
                    rec["error"] = err
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done += len(chunk)
            print(f"  [{track_dir.name}] {done}/{len(paths)} pages  {done / wall if wall else 0:.2f} p/s", flush=True)

    timing = {
        "n": done, "n_planned": len(paths), "partial": partial, "wall_s": round(wall, 3),
        "pages_per_s": done / wall if wall else None,
        "latency_n": len(latencies), "latency_p50_s": _pct(latencies, 50), "latency_p95_s": _pct(latencies, 95),
    }
    _write_json(track_dir / "timing.json", timing)
    return timing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, type=Path)
    job = json.loads(ap.parse_args(argv).job.read_text())
    out = Path(job["out"])
    spec, hardware = job["spec"], job["hardware"]
    status = {"engine": spec["id"], "status": "running", "started": time.time(),
              "python": sys.version.split()[0], "platform": platform.platform(), "tracks": {}}
    _write_json(out / "status.json", status)

    mon = ResourceMonitor()
    mon.start()
    engine = None
    try:
        engine = create(spec, hardware)
        t = time.perf_counter()
        engine.load()
        status["load_s"] = round(time.perf_counter() - t, 2)
        status["versions"] = engine.versions()

        # Warm-up page (CUDA kernels, lazy model init) is excluded from timing.
        first_manifest = Path(next(iter(job["tracks"].values())))
        t = time.perf_counter()
        engine.predict([str(first_manifest.parent / read_manifest(first_manifest)[0].image)])
        status["warmup_s"] = round(time.perf_counter() - t, 2)

        # The time budget covers inference only: slow cold starts (kernel compilation on
        # older GPUs) are reported as load_s instead of eating the page budget.
        deadline = time.time() + job["timeout_s"] if "timeout_s" in job else job["deadline"]
        for track, manifest in job["tracks"].items():
            manifest = Path(manifest)
            print(f"[{spec['id']}] track {track}", flush=True)
            try:
                status["tracks"][track] = run_track(
                    engine, read_manifest(manifest), out / track, manifest.parent,
                    job.get("latency_n", 0), deadline)
            except Exception as e:  # one bad track shouldn't sink the others...
                traceback.print_exc()
                status["tracks"][track] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                if _is_fatal(e):    # ...unless the engine itself died
                    status["error"] = status["tracks"][track]["error"]
                    break
        partial = len(status["tracks"]) < len(job["tracks"]) or any(
            t.get("partial") or "error" in t for t in status["tracks"].values())
        status["status"] = "partial" if partial else "ok"
    except Exception as e:
        traceback.print_exc()
        status.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
    finally:
        if engine is not None:
            engine.close()
        status.update(mon.stop())
        status["finished"] = time.time()
        _write_json(out / "status.json", status)
    # vLLM and friends sometimes hang on interpreter shutdown; results are already on disk.
    sys.stdout.flush()
    os._exit(0 if status["status"] != "error" else 1)


if __name__ == "__main__":
    main()
