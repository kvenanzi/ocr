"""Fake engines for exercising the worker/runner subprocess path."""
import subprocess
import sys
import time

from ocrbench.engines.base import Engine, Prediction


class EngineDeadError(RuntimeError):
    pass


class Dies(Engine):
    """Works for the warm-up and 2 more pages, then dies like vLLM after a CUDA OOM."""
    calls = 0

    def load(self):
        # an orphan child holding the stdout pipe, like vLLM's EngineCore
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])

    def predict(self, paths):
        Dies.calls += 1
        if Dies.calls > 3:
            raise EngineDeadError("EngineCore encountered an issue")
        return [Prediction("The quick brown fox") for _ in paths]


class SlowLoad(Engine):
    def load(self):
        time.sleep(3)   # longer than the 2 s page budget below

    def predict(self, paths):
        return [Prediction("The quick brown fox") for _ in paths]


class BadPage(Engine):
    """Fails on one specific page, like Tesseract erroring on one odd image."""
    native_batch = True

    def load(self):
        pass

    def predict(self, paths):
        if any(p.endswith("bad.png") for p in paths):
            raise RuntimeError("ocrmypdf exit 7: [tesseract] Error during processing.")
        return [Prediction("The quick brown fox") for _ in paths]


class SlowPages(Engine):
    """0.3 s per page: too slow to finish every track inside the budget."""

    def load(self):
        pass

    def predict(self, paths):
        time.sleep(0.3 * len(paths))
        return [Prediction("The quick brown fox") for _ in paths]
