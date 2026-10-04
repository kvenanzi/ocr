# OCR Benchmark: classic engines vs. OCR vision-language models

Which OCR should you use? It depends heavily on the material. This repo benchmarks
**classic OCR engines** (Tesseract, OCRmyPDF, RapidOCR, EasyOCR, docTR, PaddleOCR, Surya)
against **OCR-specialised VLMs** (PaddleOCR-VL, dots.mocr, DeepSeek-OCR 2, olmOCR 2, Chandra,
GLM-OCR, LightOnOCR, OvisOCR2, Granite-Docling, Surya OCR 2) and **general VLMs** (Qwen3.5,
Qwen2.5-VL, Gemma 4). Each is tested on four kinds of material with known ground truth and
timed on whatever Colab runtime you pick.

**→ Results: [results/LEADERBOARD.md](results/LEADERBOARD.md)** (built from every run pushed so far)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kvenanzi/ocr/blob/main/notebooks/ocr_benchmark_colab.ipynb)

## Quick start (Colab)

1. Open `notebooks/ocr_benchmark_colab.ipynb` in Colab (badge above).
2. *Runtime → Change runtime type*: T4, L4, G4, A100, or TPU v5e-1 / v6e-1.
3. Add the Colab secret `GITHUB_TOKEN`, a fine-grained token with *Contents: read & write* on this repo.
   `HF_TOKEN` is optional but speeds up model downloads.
4. Pick a profile and *Run all*:

| Profile | Pages | Time on A100 | Use |
|---|---|---|---|
| `smoke` | ~25 | ~15 min (mostly installs + model downloads) | check that every engine loads |
| `standard` | ~215 | ~1–1.5 h | the comparison run |
| `full` | ~1,000 | several hours | tighter numbers |

The notebook pushes `results/runs/<timestamp>_<hardware>_<profile>/` and regenerates the
leaderboard. Run it once per runtime type; each runtime adds a column to the hardware table.

## What is measured

### Four material types (tracks)

| Track | Data | Ground truth | Scored by |
|---|---|---|---|
| **books** | [FineBooks BHL IMPACT](https://huggingface.co/datasets/finebooks/bhl-impact-gt): 6 natural-history books, 1708–1913, EN/FR/DE/LA | expert transcriptions (~99.95% accurate) | CER, WER, bag-of-words recall |
| **docs** | [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench): headers/footers, multi-column, tables, tiny text, old scans | ~7k machine-checkable unit tests | test pass rate |
| **handwriting** | [IAM lines](https://huggingface.co/datasets/Teklia/IAM-line) (test split) | line transcriptions | CER, WER |
| **stress** | synthetic pages rendered from known text (prose, 2-column, tables, invoices) | the rendered text | CER at each of 11 degradation levels |

The **stress ladder** applies these to the same pages: clean → 150 dpi → 75 dpi → blur → noise →
3° rotation → 8° rotation → JPEG q10 → low contrast → photocopy (binarised, speckled,
binding shadow) → phone photo (perspective warp, uneven light). The result is a robustness
curve for each engine.

Sampling is seeded and stratified (by book, category, and layout), so every runtime scores
exactly the same pages.

### Metrics

| Group | Metric | Notes |
|---|---|---|
| Accuracy | **CER**, **WER** | *reading* mode (main), plus *diplomatic* CER (see below) |
| | bag-of-words recall / precision | order-insensitive: words read correctly, regardless of layout order |
| | over-extraction | share of output words not on the page (hallucination) |
| | olmOCR-bench pass rate | mean over categories, as the bench reports it |
| Reliability | loop rate | runaway repetition, or hitting `max_tokens` |
| | empty / failed pages | |
| Speed | load time | cold start: weights to GPU, compile, CUDA graphs |
| | latency p50 / p95 | one page at a time |
| | throughput (pages/s) | full batch, engine-native batching (vLLM continuous batching; CPU engines use all cores) |
| Resources | peak VRAM, peak RAM | sampled every 0.5 s. vLLM pre-allocates KV cache, so its VRAM reflects `gpu_memory_utilization` |
| Cost | **$ per 1,000 pages** | Colab $/hour ÷ throughput (`configs/hardware_costs.yaml`). ×1,000 gives the cost of a 1M-page collection |

### How scoring works

Engines emit very different formats: plain lines, Markdown, HTML tables, LaTeX, DocTags, or
grounding tokens. Before scoring, every output is reduced to plain text in reading order
(`ocrbench/metrics/normalize.py`), so formatting choices cost nothing.

- **Reading CER** (the headline): NFKC, long-s → s, ligatures expanded, quotes/dashes unified,
  line-end hyphenation joined, whitespace collapsed. It answers "did it read the words
  right?", which is what search, RAG, and LLM-training uses need.
- **Diplomatic CER**: no character normalisation. It matters for scholarly transcription,
  where ſ and ligatures should be preserved.
- Page CER is capped at 100%, so one runaway page can't dominate the mean. The uncapped
  micro-average is saved too (`cer_micro`).
- **Overall** = mean of the four track accuracies (1 − CER for text tracks, pass rate for docs).
- **Docs** uses a faithful reimplementation of olmOCR-bench's present/absent/order/baseline/table
  tests (`ocrbench/metrics/olmocr_tests.py`). The two math categories are excluded: they need
  KaTeX rendering in headless Chromium. Scores are therefore the *text + table subset* and
  are not directly comparable with the official leaderboard.

## Engines

All VLM prompts, sampling settings and image sizes are copied from each model card
(`configs/engines.yaml`, `ocrbench/engines/prompts.py`).

| id | Engine | Type | Params | License | Notes |
|---|---|---|---|---|---|
| `tesseract` | Tesseract 5 | classic | – | Apache-2.0 | `eng+fra+deu+lat`, LSTM, pages in parallel across cores |
| `ocrmypdf` | OCRmyPDF | classic | – | MPL-2.0 | Tesseract + deskew/clean/auto-rotate; shows what preprocessing buys |
| `rapidocr` | RapidOCR | classic | – | Apache-2.0 | PP-OCRv6 on ONNX Runtime, CPU |
| `easyocr` | EasyOCR | classic | – | Apache-2.0 | CRAFT + CRNN |
| `doctr` | docTR | classic | – | Apache-2.0 | FAST + PARSeq |
| `paddleocr` | PaddleOCR 3 | classic | – | Apache-2.0 | PP-OCRv6 det+rec, no layout model |
| `surya` | Surya 0.17 | classic | – | GPL-3.0 / OpenRAIL | last line-level release (0.22+ became a VLM; see `surya-ocr2`) |
| `paddleocr-vl` | PaddleOCR-VL 1.6 | OCR VLM | 0.96B | Apache-2.0 | whole-page `OCR:` (the official pipeline adds a layout model) |
| `ovisocr2` | OvisOCR2 | OCR VLM | 0.85B | Apache-2.0 | |
| `lightonocr2` | LightOnOCR-2 | OCR VLM | 1.0B | Apache-2.0 | |
| `glm-ocr` | GLM-OCR | OCR VLM | 1.3B | MIT | whole-page (the official SDK adds a layout model) |
| `granite-docling` | Granite-Docling | OCR VLM | 0.26B | Apache-2.0 | DocTags → Markdown via docling-core |
| `dots-mocr` | dots.mocr | OCR VLM | 3.0B | MIT | FineBooks' top model; `prompt_ocr` mode skips page headers/footers |
| `deepseek-ocr2` | DeepSeek-OCR 2 | OCR VLM | 3.4B | Apache-2.0 | grounding markdown mode, n-gram repetition guard |
| `surya-ocr2` | Surya OCR 2 | OCR VLM | 0.69B | OpenRAIL | surya 0.22's full-page prompt, served directly by vLLM |
| `chandra-ocr2` | Chandra OCR 2 | OCR VLM | 5.3B | OpenRAIL-M (modified) | |
| `olmocr2` | olmOCR 2 (FP8) | OCR VLM | 8.3B | Apache-2.0 | |
| `nanonets-ocr2` | Nanonets-OCR2 | OCR VLM | 3.75B | unclear | opt-in (`--engines`) |
| `qwen3.5-2b` / `qwen3.5-9b` | Qwen3.5 | general VLM | 2.3B / 9.7B | Apache-2.0 | thinking off |
| `qwen3-vl-8b` | Qwen3-VL | general VLM | 8.8B | Apache-2.0 | opt-in |
| `gemma4-e4b` | Gemma 4 E4B | general VLM | 8.0B | Apache-2.0 | 1120 vision tokens for OCR; needs bf16 |
| `qwen2.5-vl-3b` / `-7b` | Qwen2.5-VL | general VLM | 3.8B / 8.3B | Qwen Research / Apache-2.0 | **also runs on TPU**, for GPU-vs-TPU comparison |

### Hardware notes

- **T4** (16 GB, no bf16): models run in fp16. Granite-Docling runs in fp32 (fp16 outputs garbage).
  Models that need >15 GB or bf16 are skipped and listed in the leaderboard.
- **L4 / A100 / G4** (G4 = RTX PRO 6000 Blackwell, 96 GB): everything that fits.
- **TPU v5e-1 / v6e-1**: best effort. vLLM's TPU backend (`vllm-tpu`) runs Qwen2.5-VL 3B
  (7B needs v6e's 32 GB HBM). The classic CPU engines run on the TPU VM's host CPU. Colab
  doesn't publish TPU compute-unit rates; enter yours in the notebook's `CU_PER_HOUR` to get cost.

## Running outside Colab

```bash
pip install uv && uv pip install -e '.[dev]'
python -m ocrbench hardware                       # what was detected
python -m ocrbench run --profile smoke --dry-run  # which engines would run here
python -m ocrbench run --profile smoke --engines tesseract,rapidocr
python -m ocrbench report                         # rebuild results/LEADERBOARD.md
pytest
```

Each engine family gets its own virtualenv (`ocrbench/envs.py`), created on first use:
`classic`, `paddle`, `surya`, `vllm`, `vllm-tpu`. This keeps vLLM, PaddlePaddle, and old Surya
from fighting over torch/transformers/pillow versions. Every engine runs in a subprocess
with a time budget, so a crash or hang costs one row, not the run. `--no-venv` runs engines
in the current interpreter (handy for local development).

### Adding an engine

- **A VLM that vLLM supports:** add an entry to `configs/engines.yaml` with
  `impl: vllm_engine` and its card's prompt and sampling. No code is needed.
- **Anything else:** add `ocrbench/engines/<name>.py` with a class `ENGINE(Engine)`
  implementing `load()` and `predict(paths) -> list[Prediction]`, then register it.

## Caveats (read before quoting numbers)

- **Ground truth includes page furniture.** Books ground truth contains running heads and page
  numbers. Models that drop headers by design (dots.mocr in `prompt_ocr` mode, olmOCR) lose some
  recall on that track. Conversely, olmOCR-bench's *headers_footers* tests reward dropping them.
- **Whole-page vs. pipeline.** PaddleOCR-VL and GLM-OCR are run on whole pages. Their official
  products first run a layout model and OCR each region, which likely scores higher.
- **Classic engines read in raster order.** On two-column pages they interleave lines across
  columns, which CER punishes heavily. Bag-of-words recall shows whether the words themselves
  were read.
- **Costs are approximate.** They come from third-party-measured Colab compute-unit rates
  (`configs/hardware_costs.yaml`, March 2026) at $0.0999/CU. Self-hosted or spot GPUs are far cheaper.
- **Sampling follows each model card.** Some cards recommend temperature > 0 (LightOnOCR 0.2,
  dots 0.1, olmOCR 0.1). Seeds are fixed, but results can vary slightly between runs.
- **Small samples.** The standard profile uses ~50 pages per track. Differences under ~1 CER point
  are within noise.
- **VLM speeds from runs before commit `670efe5`+1 had vLLM's prefix cache on.** The first 8 pages
  of each batch pass repeated the single-page latency pass and could reuse its work. On A100 and L4
  the first batch was not consistently faster than the second (within page-to-page noise). On T4,
  where reading the image dominates, prefill-heavy models (dots.mocr) look faster than they are.
  A run's code version is in its `env.json` (`git_sha`).

## Repo layout

```
configs/      engines.yaml (registry) · profiles.yaml · hardware_costs.yaml
ocrbench/     cli · runner (gating, subprocesses) · worker (timing, memory) · envs (venvs)
              data/ (track loaders, synthetic renderer) · engines/ · metrics/ · score · report · publish
notebooks/    ocr_benchmark_colab.ipynb (generated by tools/build_notebook.py)
results/      LEADERBOARD.md · leaderboard.csv · hardware.csv · report.html · charts/ · runs/
tests/        pytest suite (metrics, olmOCR tests, synthetic data, engine wiring with mocked vLLM)
```

## Credits

FineBooks / IMPACT / BHL ground truth (CC-BY 3.0), olmOCR-bench (ODC-BY, Ai2), IAM handwriting
database (via Teklia, MIT), and every model and engine author listed above. The idea of
reading vs. diplomatic CER and loop rate comes from the
[FineBooks historical-books OCR leaderboard](https://huggingface.co/blog/finebooks/historical-books-ocr-leaderboard).
