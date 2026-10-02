"""docTR (Mindee): FAST/DBNet detection + CRNN/PARSeq recognition, PyTorch."""
from __future__ import annotations

from .base import Engine, Prediction, pkg_version


class ENGINE(Engine):
    native_batch = True
    batch_size = 8

    def load(self):
        import torch
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        self.DocumentFile = DocumentFile
        dev = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        self.model = ocr_predictor(det_arch=self.params.get("det_arch", "fast_base"),
                                   reco_arch=self.params.get("reco_arch", "parseq"),
                                   pretrained=True, assume_straight_pages=False,
                                   det_bs=4, reco_bs=256).to(dev)

    def predict(self, image_paths):
        doc = self.model(self.DocumentFile.from_images(list(image_paths)))
        return [Prediction(page.render()) for page in doc.pages]

    def versions(self):
        return {"python-doctr": pkg_version("python-doctr"), "torch": pkg_version("torch")}
