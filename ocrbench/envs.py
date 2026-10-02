"""One virtualenv per engine family, so conflicting dependencies never meet.

  classic   Tesseract/OCRmyPDF/RapidOCR/EasyOCR/docTR. Reuses Colab's preinstalled
            torch via --system-site-packages, so it installs in about a minute.
  paddle    PaddlePaddle + PaddleOCR 3.x
  surya     Surya 0.17 (classic line-level OCR; needs pillow<11, transformers<5)
  vllm      vLLM for every OCR VLM; `uv --torch-backend=auto` picks the PyTorch
            CUDA build that matches the installed driver
  vllm-tpu  vLLM TPU backend (tpu-inference / JAX)

Envs are created lazily the first time an engine needs them, and reused as long
as the recipe hasn't changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import config
from .hardware import detect

VLLM_VERSION = os.environ.get("OCRBENCH_VLLM_VERSION", "0.30.0")

RECIPES: dict[str, dict] = {
    "classic": {
        "system_site_packages": True,
        "steps": [["pytesseract>=0.3.13", "ocrmypdf>=17", "pillow>=12", "rapidocr>=3.9", "onnxruntime",
                   "easyocr>=1.7.2", "python-doctr>=1.1"]],
    },
    "paddle": {
        "steps_gpu": [["paddlepaddle-gpu==3.3.1", "--index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"],
                      ["paddleocr==3.7.0"]],
        "steps_cpu": [["paddlepaddle==3.3.1", "--index-url", "https://www.paddlepaddle.org.cn/packages/stable/cpu/"],
                      ["paddleocr==3.7.0"]],
    },
    "surya": {
        "system_site_packages": True,
        "steps": [["surya-ocr==0.17.1", "transformers>=4.56.1,<5"]],
    },
    "vllm": {
        "uv": True,
        "steps": [[f"vllm=={VLLM_VERSION}", "docling-core>=2.40", "pillow", "psutil", "--torch-backend=auto"]],
    },
    "vllm-tpu": {
        "uv": True,
        "steps": [[f"vllm-tpu=={VLLM_VERSION}", "pillow", "psutil"]],
    },
}


def _recipe(family: str) -> dict:
    r = dict(RECIPES[family])
    if "steps" not in r:
        r["steps"] = r["steps_gpu"] if detect().kind == "gpu" else r["steps_cpu"]
    return r


def _run(cmd: list[str], log: Path | None) -> None:
    with open(log, "a") if log else open(os.devnull, "w") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        tail = log.read_text(errors="replace")[-1500:] if log else ""
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd[:6])} ...\n{tail}")


def python_for(family: str, log: Path | None = None) -> str:
    """Return the interpreter for an engine family, creating the venv if needed."""
    recipe = _recipe(family)
    env_dir = config.ENVS_DIR / family
    py = env_dir / "bin" / "python"
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    marker = env_dir / ".ocrbench_ready"
    if py.exists() and marker.exists() and marker.read_text().strip() == digest:
        return str(py)

    print(f"  creating env '{family}' (one-time, a few minutes) -> {env_dir}", flush=True)
    shutil.rmtree(env_dir, ignore_errors=True)
    uv = shutil.which("uv")
    if recipe.get("uv") and uv:
        _run([uv, "venv", "--python", sys.executable, "--seed", str(env_dir)], log)
        for step in recipe["steps"]:
            _run([uv, "pip", "install", "--python", str(py), *step], log)
    else:
        args = [sys.executable, "-m", "venv", str(env_dir)]
        if recipe.get("system_site_packages"):
            args.insert(3, "--system-site-packages")
        _run(args, log)
        for step in recipe["steps"]:
            # torch-backend is a uv-only flag
            step = [s for s in step if not s.startswith("--torch-backend")]
            _run([str(py), "-m", "pip", "install", "-q", *step], log)
    marker.write_text(digest)
    return str(py)


def worker_env(spec: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    env.setdefault("VLLM_CONFIGURE_LOGGING", "1")
    env.update({k: str(v) for k, v in spec.get("env", {}).items()})
    return env
