"""Plan and execute a benchmark run: which engines fit this hardware, then one subprocess each."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config, envs
from .hardware import HardwareInfo, detect


def load_registry() -> list[dict]:
    return config.load_yaml("engines.yaml")["engines"]


def family_for(spec: dict, hw: HardwareInfo) -> str:
    """Env family, optionally overridden per accelerator (e.g. family_tpu: vllm-tpu)."""
    return spec.get(f"family_{hw.kind}", spec["family"])


def gate(spec: dict, hw: HardwareInfo) -> str | None:
    """Return a skip reason, or None if the engine can run on this hardware."""
    req = spec.get("requires", {})
    kinds = req.get("hardware", ["gpu", "cpu", "tpu"])
    if hw.kind not in kinds:
        return f"unsupported_hardware: needs {'/'.join(kinds)}"
    if hw.kind == "gpu":
        min_cc = max(req.get("min_compute_capability", 0), 7.0 if family_for(spec, hw) == "vllm" else 0)
        if min_cc > hw.compute_capability:
            return f"gpu_too_old: needs sm{min_cc:g}, have sm{hw.compute_capability:g}"
        if req.get("min_vram_gb", 0) > hw.vram_gb:
            return f"insufficient_vram: needs {req['min_vram_gb']} GB, have {hw.vram_gb}"
        if req.get("bf16") and not hw.bf16:
            return "needs_bf16: GPU has no bf16 support (e.g. T4)"
    if hw.kind == "tpu" and req.get("min_hbm_gb", 0) > hw.vram_gb:
        return f"insufficient_hbm: needs {req['min_hbm_gb']} GB, have {hw.vram_gb}"
    return None


def plan(hw: HardwareInfo, include: list[str] | None = None, exclude: list[str] | None = None) -> list[dict]:
    rows = []
    for spec in load_registry():
        if include and spec["id"] not in include:
            continue
        reason = None
        if exclude and spec["id"] in exclude:
            reason = "excluded"
        elif not include and spec.get("default") is False:
            reason = "not_default: opt in with --engines"
        reason = reason or gate(spec, hw)
        rows.append({"id": spec["id"], "family": family_for(spec, hw), "run": reason is None,
                     "skip_reason": reason, "spec": spec})
    return rows


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(config.REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _tee(proc: subprocess.Popen, log_path: Path, prefix: str) -> threading.Thread:
    def pump():
        with open(log_path, "a", encoding="utf-8") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                # vLLM is chatty; only echo our own progress lines and errors to the notebook.
                if line.startswith(("[", "  [")) or "Error" in line or "Traceback" in line:
                    print(f"{prefix}{line}", end="", flush=True)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


def _gpu_used_mib() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return sum(int(x) for x in out.split())
    except Exception:
        return None


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the worker and everything it spawned. A worker that exits without shutting down
    vLLM leaves an orphaned EngineCore holding most of the GPU, and every later engine fails."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _wait_for_gpu_release(baseline: int | None, timeout_s: float = 90) -> None:
    if baseline is None:
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_used_mib()
        if used is None or used <= baseline + 512:
            return
        time.sleep(2)
    print(f"  warning: GPU memory still in use after engine exit ({_gpu_used_mib()} MiB)", flush=True)


def run_engine(spec: dict, hw: HardwareInfo, tracks: dict[str, Path], out: Path,
               latency_n: int, timeout_min: float, use_venvs: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        python = envs.python_for(family_for(spec, hw), log=out / "install.log") if use_venvs else sys.executable
    except Exception as e:
        status = {"engine": spec["id"], "status": "error", "error": f"env install failed: {e}"}
        (out / "status.json").write_text(json.dumps(status, indent=2))
        return status
    install_s = time.time() - started

    job = {"spec": spec, "hardware": hw.to_dict(), "out": str(out), "latency_n": latency_n,
           "tracks": {k: str(v) for k, v in tracks.items()}, "deadline": time.time() + timeout_min * 60}
    (out / "job.json").write_text(json.dumps(job, indent=2))
    env = envs.worker_env(spec)
    gpu_baseline = _gpu_used_mib()
    # Own process group, so vLLM's EngineCore child processes can be killed with the worker.
    proc = subprocess.Popen([python, "-u", "-m", "ocrbench.worker", "--job", str(out / "job.json")],
                            cwd=config.REPO_ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    pump = _tee(proc, out / "worker.log", prefix="    ")
    hard_limit = timeout_min * 60 + 600   # grace for model download/load overrun
    try:
        proc.wait(timeout=hard_limit)
    except subprocess.TimeoutExpired:
        pass
    _kill_group(proc)
    pump.join(timeout=30)
    _wait_for_gpu_release(gpu_baseline)

    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"engine": spec["id"]}
    if status.get("status") in (None, "running"):
        status["status"] = "partial" if status.get("tracks") else "error"
        status["error"] = status.get("error") or f"worker exited with code {proc.returncode} (killed or crashed)"
    status["install_s"] = round(install_s, 1)
    status["wall_s"] = round(time.time() - started, 1)
    status_path.write_text(json.dumps(status, indent=2, default=str))
    return status


def purge_model_cache(spec: dict) -> None:
    """Delete a VLM's downloaded weights (Colab disks can't hold every model at once)."""
    model = spec.get("params", {}).get("model")
    if not model or os.environ.get("OCRBENCH_PURGE_MODELS") != "1":
        return
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    target = hub / f"models--{model.replace('/', '--')}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        print(f"  purged {target.name} from the HF cache", flush=True)


def run(profile_name: str, tracks: list[str] | None = None, include=None, exclude=None,
        run_id: str | None = None, dry_run: bool = False, use_venvs: bool = True,
        timeout_min: float | None = None, cu_per_hour: float | None = None) -> Path | None:
    from .data import prepare_tracks

    prof = config.profile(profile_name)
    hw = detect()
    rows = plan(hw, include, exclude)
    tracks = tracks or list(config.TRACKS)

    print(f"Hardware: {hw.tag} ({hw.accelerator or hw.cpu_model}), {hw.vram_gb} GB, {hw.cpu_count} CPUs")
    for r in rows:
        mark = "RUN " if r["run"] else "skip"
        print(f"  {mark} {r['id']:<22} {r['family']:<9} {r['skip_reason'] or ''}")
    if dry_run:
        return None

    manifests = prepare_tracks(profile_name, tracks)
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{hw.tag}_{profile_name}"
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "profile": prof, "tracks": tracks, "hardware": hw.to_dict(),
            "git_sha": _git_sha(), "started": time.time(), "cu_per_hour": cu_per_hour,
            "plan": [{k: r[k] for k in ("id", "family", "run", "skip_reason")} for r in rows]}
    (run_dir / "env.json").write_text(json.dumps(meta, indent=2))

    for r in [r for r in rows if r["run"]]:
        print(f"\n=== {r['id']} ===", flush=True)
        status = run_engine(r["spec"], hw, manifests, run_dir / r["id"], prof["latency_n"],
                            timeout_min or prof["timeout_min"], use_venvs)
        print(f"  -> {status.get('status')}  load {status.get('load_s', '?')}s  "
              f"{status.get('error', '')}", flush=True)
        purge_model_cache(r["spec"])

    meta["finished"] = time.time()
    (run_dir / "env.json").write_text(json.dumps(meta, indent=2))
    return run_dir
