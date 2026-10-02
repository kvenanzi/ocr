"""OCRmyPDF: Tesseract plus a preprocessing pipeline (deskew, clean, auto-rotate).

Comparing it with plain Tesseract shows how much classic image cleanup buys you.
Each page is converted to a PDF with a text layer; we score the sidecar text.
Run through the CLI, one page per process, across CPU cores.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from ._parallel import cpu_workers, pmap
from .base import Engine, Prediction, pkg_version


class ENGINE(Engine):
    native_batch = True

    def load(self):
        self.lang = self.params.get("lang", "eng+fra+deu+lat")
        self.flags = self.params.get("flags", ["--deskew", "--clean", "--rotate-pages"])
        self.workers = cpu_workers(self.params)
        self.batch_size = self.workers * 2
        self.tmp = Path(tempfile.mkdtemp(prefix="ocrmypdf_"))
        subprocess.run(["ocrmypdf", "--version"], check=True, capture_output=True)

    def _one(self, path: str) -> Prediction:
        stem = self.tmp / f"{Path(path).stem}_{os.getpid()}_{id(path)}"
        cmd = ["ocrmypdf", "-q", "--force-ocr", "--image-dpi", str(self.params.get("image_dpi", 300)),
               "-l", self.lang, "--jobs", "1", "--output-type", "pdf", *self.flags,
               "--sidecar", f"{stem}.txt", path, f"{stem}.pdf"]
        res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        if res.returncode != 0:
            raise RuntimeError(f"ocrmypdf exit {res.returncode}: {res.stderr[-400:]}")
        text = Path(f"{stem}.txt").read_text(errors="replace")
        for ext in (".txt", ".pdf"):
            Path(f"{stem}{ext}").unlink(missing_ok=True)
        # sidecar marks pages without OCR text with a form feed / "[OCR skipped on page]"
        return Prediction(text.replace("\f", "\n").replace("[OCR skipped on page(s) 1]", ""))

    def predict(self, image_paths):
        return pmap(self._one, image_paths, self.workers)

    def versions(self):
        return {"ocrmypdf": pkg_version("ocrmypdf")}
