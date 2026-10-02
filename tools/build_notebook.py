"""Generate notebooks/ocr_benchmark_colab.ipynb (edit here, then re-run this script)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# OCR Benchmark: classic engines vs. OCR VLMs

Benchmarks **Tesseract, OCRmyPDF, RapidOCR, EasyOCR, docTR, PaddleOCR, Surya** against **OCR vision-language models**
(PaddleOCR-VL, dots.mocr, DeepSeek-OCR 2, olmOCR 2, Chandra, GLM-OCR, LightOnOCR, OvisOCR2, Granite-Docling, Qwen, Gemma)
on four material types with known ground truth:

| Track | Data | Metric |
|---|---|---|
| Historical books | FineBooks BHL IMPACT (expert transcriptions, EN/FR/DE/LA, 1708-1913) | CER / WER |
| Modern documents | olmOCR-bench (text + table subset) | unit-test pass rate |
| Handwriting | IAM lines | CER / WER |
| Stress test | synthetic pages, 11 degradation levels | CER per level |

It also measures load time, latency, throughput, peak memory, and **$ per 1,000 pages** on this runtime.

**How to use**
1. *Runtime → Change runtime type*: pick a GPU (T4 / L4 / G4 / A100) or TPU (v5e-1 / v6e-1).
2. Add Colab secrets (key icon on the left): `GITHUB_TOKEN` (repo read/write) and optionally `HF_TOKEN` (faster downloads).
3. Adjust the form below, then *Runtime → Run all*.
4. Results are pushed to `results/runs/<run>` and the leaderboard `results/LEADERBOARD.md` is regenerated.

Run it once per runtime type: the leaderboard gains a column per hardware.
"""),
code("""#@title Settings { display-mode: "form" }
REPO = "kvenanzi/ocr"  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}
PROFILE = "standard"  #@param ["smoke", "standard", "full"]
#@markdown Comma-separated engine ids. Leave empty for all default engines that fit this hardware.
ENGINES = ""  #@param {type:"string"}
EXCLUDE = ""  #@param {type:"string"}
#@markdown Tracks to run (empty = books, docs, handwriting, stress).
TRACKS = ""  #@param {type:"string"}
PUSH_RESULTS = True  #@param {type:"boolean"}
#@markdown Colab compute units per hour for this runtime (see the Resources panel). 0 = use `configs/hardware_costs.yaml`. **Set this on TPU runtimes**; their rate isn't in the table.
CU_PER_HOUR = 0.0  #@param {type:"number"}
#@markdown Per-engine time budget in minutes (0 = profile default).
TIMEOUT_MIN = 0  #@param {type:"number"}
"""),
md("## 1. Setup: clone the repo and install the harness"),
code("""import os, subprocess, sys
from google.colab import userdata

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

os.environ["GITHUB_TOKEN"] = secret("GITHUB_TOKEN") or ""
if secret("HF_TOKEN"):
    os.environ["HF_TOKEN"] = secret("HF_TOKEN")
assert os.environ["GITHUB_TOKEN"] or not PUSH_RESULTS, "Add a GITHUB_TOKEN Colab secret (or untick PUSH_RESULTS)"

os.environ["OCRBENCH_DATA"] = "/content/ocrbench_data"
os.environ["OCRBENCH_ENVS"] = "/content/ocrbench_envs"
os.environ["OCRBENCH_PURGE_MODELS"] = "1"   # delete each VLM's weights after its run to save disk
REPO_DIR = "/content/ocr"

def git(*args, auth=False):
    cmd = ["git"]
    if auth and os.environ["GITHUB_TOKEN"]:
        import base64
        b = base64.b64encode(f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()).decode()
        cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {b}"]
    r = subprocess.run(cmd + list(args), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])   # never print cmd: it holds the token header
    return r.stdout

if os.path.isdir(f"{REPO_DIR}/.git"):
    git("-C", REPO_DIR, "fetch", "origin", BRANCH, auth=True)
    git("-C", REPO_DIR, "checkout", BRANCH)
    git("-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}")
else:
    git("clone", "--branch", BRANCH, f"https://github.com/{REPO}.git", REPO_DIR, auth=True)
os.chdir(REPO_DIR)
print(git("log", "-1", "--oneline"))

!pip -q install uv
!uv pip -q install --system -e .
"""),
md("""## 2. System packages

Tesseract **5** (Colab's Ubuntu ships 4.1 by default) with English/French/German/Latin data, plus `unpaper` and Ghostscript for OCRmyPDF."""),
code("""%%bash
set -e
if ! tesseract --version 2>/dev/null | head -1 | grep -q "tesseract 5"; then
  add-apt-repository -y ppa:alex-p/tesseract-ocr5 > /dev/null 2>&1 || true
  apt-get -qq update > /dev/null
fi
apt-get -qq install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-lat \\
    tesseract-ocr-osd unpaper ghostscript > /dev/null
tesseract --version | head -1
"""),
md("## 3. Hardware and engine plan\n\nEngines that don't fit this runtime (VRAM, bf16, TPU support) are skipped and listed in the leaderboard."),
code("""import shlex
def ocrbench_args():
    a = ["--profile", PROFILE]
    if ENGINES.strip(): a += ["--engines", ENGINES.strip()]
    if EXCLUDE.strip(): a += ["--exclude", EXCLUDE.strip()]
    if TRACKS.strip(): a += ["--tracks", TRACKS.strip()]
    if CU_PER_HOUR: a += ["--cu-per-hour", str(CU_PER_HOUR)]
    if TIMEOUT_MIN: a += ["--timeout-min", str(TIMEOUT_MIN)]
    return " ".join(shlex.quote(x) for x in a)

!python -m ocrbench hardware
!python -m ocrbench run {ocrbench_args()} --dry-run
"""),
md("## 4. Prepare datasets\n\nDownloads only the sampled pages (deterministic seed), so every runtime scores exactly the same pages."),
code("""!python -m ocrbench prepare --profile {PROFILE} {("--tracks " + TRACKS) if TRACKS.strip() else ""}"""),
md("""## 5. Run the benchmark

Each engine runs in its own subprocess and virtualenv (`classic`, `paddle`, `surya`, `vllm`, `vllm-tpu`, created on first use).
A crash or timeout in one engine is recorded and the run moves on. Full logs: `results/runs/<run>/<engine>/worker.log`."""),
code("""import json, time
from ocrbench.hardware import detect
RUN_ID = f"{time.strftime('%Y%m%d-%H%M%S')}_{detect().tag}_{PROFILE}"
print("run id:", RUN_ID)
!python -m ocrbench run {ocrbench_args()} --run-id {RUN_ID}
"""),
md("## 6. Results for this run"),
code("""import pandas as pd
from IPython.display import display, Image, Markdown
from ocrbench import config
from ocrbench.score import score_run

run_dir = config.RUNS_DIR / RUN_ID
s = score_run(run_dir)
cols = ["engine", "track", "status", "n", "accuracy", "cer", "wer", "bow_recall", "loop_rate",
        "pages_per_s", "latency_p50_s", "usd_per_1k_pages", "load_s", "peak_vram_gb"]
view = s[[c for c in cols if c in s]].copy()
fmt = {c: "{:.1%}" for c in ["accuracy", "cer", "wer", "bow_recall", "loop_rate"] if c in view}
fmt.update({"pages_per_s": "{:.2f}", "latency_p50_s": "{:.2f}", "usd_per_1k_pages": "${:.3f}", "load_s": "{:.0f}", "peak_vram_gb": "{:.1f}"})
display(view.sort_values(["track", "accuracy"], ascending=[True, False]).style.format(fmt, na_rep="–").hide(axis="index"))
"""),
md("## 7. Rebuild the leaderboard (all runs, all hardware)"),
code("""from ocrbench.report import build
path = build()
for chart in ["pareto", "tracks", "robustness"]:
    p = config.RESULTS_DIR / "charts" / f"{chart}.png"
    if p.exists():
        display(Image(filename=str(p)))
display(Markdown(path.read_text()))
"""),
md("## 8. Push results to GitHub"),
code("""if PUSH_RESULTS:
    !python -m ocrbench publish {RUN_ID} --branch {BRANCH}
else:
    print("PUSH_RESULTS is off. Results are in", run_dir)
"""),
md("""### Troubleshooting
* **An engine shows `error`**: open `results/runs/<run>/<engine>/worker.log` (and `install.log` for env problems). Re-run just that engine with `ENGINES = "<id>"`.
* **Out of disk**: each VLM's weights are deleted after it runs (`OCRBENCH_PURGE_MODELS=1`). If the vLLM env itself fills the disk, exclude the largest models (`qwen3.5-9b`, `gemma4-e4b`, `qwen2.5-vl-7b`).
* **Session limits on T4**: use `PROFILE = "smoke"` first, or split the engine list across sessions. Each session pushes its own run.
"""),
]
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm"},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent.parent / "notebooks" / "ocr_benchmark_colab.ipynb"
nbf.write(nb, out)
print("wrote", out)
