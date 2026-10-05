"""RapidOCR: PaddleOCR's PP-OCR models exported to ONNX Runtime. No Paddle, no torch.

The lightest engine in the benchmark: pip install and go, CPU-friendly.
Default models are PP-OCRv6 "small" multilingual det+rec.
"""
from __future__ import annotations

from ._parallel import cpu_workers, pmap
from .base import Engine, Prediction, pkg_version


class ENGINE(Engine):
    native_batch = True

    def load(self):
        from rapidocr import RapidOCR

        # Benchmarked as a CPU engine (its main appeal); onnxruntime-gpu wheels are CUDA-version fragile.
        # A few concurrent pages, each with a share of the cores (unbounded onnxruntime
        # threads per call oversubscribe the CPU and run slower).
        cpus = cpu_workers(self.params)
        self.workers = min(4, cpus)
        params = {"EngineConfig.onnxruntime.intra_op_num_threads": max(1, cpus // self.workers),
                  **self.params.get("rapidocr_params", {})}
        self.ocr = RapidOCR(params=params)
        self.batch_size = self.workers * 2

    def _one(self, path: str) -> Prediction:
        out = self.ocr(path)
        return Prediction("\n".join(out.txts or ()))

    def predict(self, image_paths):
        return pmap(self._one, image_paths, self.workers)

    def versions(self):
        return {"rapidocr": pkg_version("rapidocr"), "onnxruntime": pkg_version("onnxruntime") or pkg_version("onnxruntime-gpu")}
