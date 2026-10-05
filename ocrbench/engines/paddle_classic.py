"""PaddleOCR 3.x classic pipeline (PP-OCRv6 det + rec by default) on PaddlePaddle.

This is the "PaddleOCR" most people mean: text detection + line recognition,
no layout model. Line order is detection order (roughly top-to-bottom).
"""
from __future__ import annotations

from .base import Engine, Prediction, pkg_version


class ENGINE(Engine):
    native_batch = True
    batch_size = 8

    def load(self):
        from paddleocr import PaddleOCR

        kw = dict(self.params.get("paddle_kwargs", {}))
        kw.setdefault("use_doc_orientation_classify", False)
        kw.setdefault("use_doc_unwarping", False)
        kw.setdefault("use_textline_orientation", False)
        if not self.use_gpu:
            # Paddle 3.x's oneDNN path fails on PP-OCR models ("ConvertPirAttribute2RuntimeAttribute")
            kw.setdefault("enable_mkldnn", False)
        self.ocr = PaddleOCR(device="gpu:0" if self.use_gpu else "cpu", **kw)

    def predict(self, image_paths):
        results = self.ocr.predict(list(image_paths))
        return [Prediction("\n".join(r["rec_texts"])) for r in results]

    def versions(self):
        return {"paddleocr": pkg_version("paddleocr"), "paddlepaddle": pkg_version("paddlepaddle-gpu") or pkg_version("paddlepaddle")}
