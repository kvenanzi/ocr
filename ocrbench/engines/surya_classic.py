"""Surya 0.17 (datalab-to): the classic line-level detector + recogniser.

Surya 0.22+ moved to a VLM served by vLLM in Docker, which won't run in
Colab; that newer model is benchmarked separately through the vLLM engine. This
pins the last classic release, the "older tech" in the comparison.
"""
from __future__ import annotations

from PIL import Image

from .base import Engine, Prediction, pkg_version


class ENGINE(Engine):
    native_batch = True
    batch_size = 8

    def load(self):
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self.det = DetectionPredictor()
        self.rec = RecognitionPredictor(FoundationPredictor())

    def predict(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        results = self.rec(images, det_predictor=self.det)
        return [Prediction("\n".join(line.text for line in r.text_lines)) for r in results]

    def versions(self):
        return {"surya-ocr": pkg_version("surya-ocr"), "torch": pkg_version("torch")}
