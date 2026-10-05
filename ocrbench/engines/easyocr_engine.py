"""EasyOCR (JaidedAI): CRAFT detector + CRNN recogniser, PyTorch, GPU or CPU."""
from __future__ import annotations

from .base import Engine, Prediction, pkg_version


class ENGINE(Engine):
    def load(self):
        import easyocr

        self.reader = easyocr.Reader(self.params.get("langs", ["en", "fr", "de", "la"]), gpu=self.use_gpu)

    def predict(self, image_paths):
        out = []
        for p in image_paths:   # paths, not PIL: EasyOCR only accepts JPEG PIL images
            lines = self.reader.readtext(p, detail=0, paragraph=True, batch_size=16)
            out.append(Prediction("\n".join(lines)))
        return out

    def versions(self):
        return {"easyocr": pkg_version("easyocr"), "torch": pkg_version("torch")}
