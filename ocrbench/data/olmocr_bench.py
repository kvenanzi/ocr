"""Modern documents track: olmOCR-bench (allenai/olmOCR-bench, ODC-BY).

olmOCR-bench scores documents with small machine-checkable unit tests instead of
a reference transcription: "this sentence is present", "this header is absent",
"A comes before B", "cell X sits under heading Y". We sample whole PDFs,
stratified by category, and keep all of each PDF's tests.

The two math categories are excluded: scoring them needs KaTeX rendering in
headless Chromium. Results are therefore the *text + table subset* of the bench
and are not directly comparable with the official leaderboard's overall score.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from .base import Sample, stratified_sample, write_manifest

REPO = "allenai/olmOCR-bench"
CATEGORIES = ["headers_footers", "long_tiny_text", "multi_column", "old_scans", "table_tests"]
LONGEST_SIDE = 2048   # olmOCR's own renderer uses 2048 px for API models


def render_pdf(pdf_path: str, out_png: Path, longest: int = LONGEST_SIDE) -> None:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[0]
    scale = longest / max(page.get_size())
    page.render(scale=scale).to_pil().convert("RGB").save(out_png)
    pdf.close()


def prepare(out_dir: Path, n: int, seed: int) -> list[Sample]:
    tests_by_pdf: dict[str, list[dict]] = defaultdict(list)
    for cat in CATEGORIES:
        path = hf_hub_download(REPO, f"bench_data/{cat}.jsonl", repo_type="dataset")
        with open(path) as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    t["category"] = cat
                    tests_by_pdf[t["pdf"]].append(t)

    groups: dict[str, list[str]] = defaultdict(list)
    for pdf, tests in tests_by_pdf.items():
        groups[tests[0]["category"]].append(pdf)
    picked = stratified_sample({k: sorted(v) for k, v in groups.items()}, n, random.Random(seed))

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    samples = []
    for pdf in sorted(picked):
        local = hf_hub_download(REPO, f"bench_data/pdfs/{pdf}", repo_type="dataset")
        sid = pdf.removesuffix(".pdf").replace("/", "__")
        rel = f"images/{sid}.png"
        render_pdf(local, out_dir / rel)
        tests = tests_by_pdf[pdf]
        samples.append(Sample(id=sid, track="docs", image=rel,
                              meta={"category": tests[0]["category"], "pdf": pdf, "tests": tests}))
    write_manifest(out_dir, samples)
    return samples
