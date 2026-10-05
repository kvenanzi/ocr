"""Tesseract 5 (LSTM engine) via pytesseract.

Tesseract is single-threaded per page, so for throughput we OCR several pages at
once across CPU cores (OMP_THREAD_LIMIT=1 per process), which is how you would
run it over a large collection.
"""
from __future__ import annotations

import os
import subprocess

from ._parallel import cpu_workers, pmap
from .base import Engine, Prediction


class ENGINE(Engine):
    native_batch = True

    def load(self):
        import pytesseract

        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        self.pt = pytesseract
        self.lang = self.params.get("lang", "eng+fra+deu+lat")
        self.config = self.params.get("config", "--oem 1 --psm 3")
        self.workers = cpu_workers(self.params)
        self.batch_size = self.workers * 2
        missing = set(self.lang.split("+")) - set(pytesseract.get_languages(config=""))
        if missing:
            raise RuntimeError(f"tesseract language data missing: {missing} (apt install tesseract-ocr-<lang>)")

    def _one(self, path: str) -> Prediction:
        return Prediction(self.pt.image_to_string(path, lang=self.lang, config=self.config))

    def predict(self, image_paths):
        return pmap(self._one, image_paths, self.workers)

    def versions(self):
        out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout
        return {"tesseract": out.splitlines()[0] if out else "?"}
