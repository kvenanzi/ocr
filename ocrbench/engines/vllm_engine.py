"""Generic vision-language-model OCR engine on vLLM (GPU, or TPU via vllm-tpu).

Every VLM in configs/engines.yaml uses this class. The model-specific bits come
from the registry: prompt, where the image goes, sampling, LLM kwargs, image
sizing, and output post-processing. All were taken from each model card.
Pages are sent in batches so vLLM's continuous batching can work: that is the
"throughput" number. Single-page calls give the "latency" number.
"""
from __future__ import annotations

import importlib
import re

from PIL import Image

from .base import Engine, Prediction, pkg_version
from .prompts import PROMPTS


def _resolve(dotted: str):
    module, _, attr = dotted.partition(":")
    return getattr(importlib.import_module(module), attr)


# ------------------------------------------------------------------ post-processing

def _strip_think(text, **_):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def _strip_front_matter(text, **_):
    return re.sub(r"\A\s*---\n.*?\n---\s*\n", "", text, count=1, flags=re.S)


def _drop_bbox_img_tags(text, **_):
    return re.sub(r'<img src="images/bbox_\d+_\d+_\d+_\d+\.jpg" ?/>', "", text)


def _drop_img_descriptions(text, **_):
    return re.sub(r"<img>.*?</img>", "", text, flags=re.S)


def _deepseek(text, **_):
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", text, flags=re.S)
    return text.replace("<｜end▁of▁sentence｜>", "")


def _trim_repeats(text, min_text_len=8000, max_period=200, min_repeat_chars=100, min_repeat_times=5, **_):
    """OvisOCR2's reference cleanup: cut a long repeated tail down to one copy."""
    n = len(text)
    if n < min_text_len:
        return text
    for unit in range(1, min(max_period, n - 1) + 1):
        if text[n - 1] != text[n - 1 - unit]:
            continue
        match, idx = 1, n - 2
        while idx >= unit and text[idx] == text[idx - unit]:
            match += 1
            idx -= 1
        total = match + unit
        if total // unit >= min_repeat_times and total >= min_repeat_chars:
            return text[: n - total + unit] + text[n - total % unit:]
    return text


def _doctags(text, image=None, **_):
    """granite-docling emits DocTags; convert with docling-core."""
    try:
        from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

        dt = DocTagsDocument.from_doctags_and_image_pairs([text], [image])
        return DoclingDocument.load_from_doctags(dt, document_name="page").export_to_markdown()
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)


POSTPROCESS = {f.__name__.lstrip("_"): f for f in (
    _strip_think, _strip_front_matter, _drop_bbox_img_tags, _drop_img_descriptions, _deepseek, _trim_repeats, _doctags)}


# ------------------------------------------------------------------ engine

def _fit(img: Image.Image, max_side: int | None, max_pixels: int | None) -> Image.Image:
    w, h = img.size
    scale = 1.0
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
    if max_pixels and w * h * scale * scale > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


class ENGINE(Engine):
    native_batch = True

    def load(self):
        from vllm import LLM, SamplingParams

        p = self.params
        self.batch_size = p.get("batch_size", 32)
        kw = {
            "model": p["model"],
            "trust_remote_code": p.get("trust_remote_code", False),
            "max_model_len": p.get("max_model_len", 16384),
            "gpu_memory_utilization": p.get("gpu_memory_utilization", 0.85),
            "limit_mm_per_prompt": p.get("limit_mm_per_prompt", {"image": 1}),
            "seed": 0,
        }
        if p.get("revision"):
            kw["revision"] = p["revision"]
        kw.update(p.get("llm_kwargs", {}))
        if self.hw.get("kind") == "gpu" and not self.hw.get("bf16"):
            kw["dtype"] = p.get("dtype_no_bf16", "float16")   # T4 (sm75) has no bf16
        if self.hw.get("kind") == "tpu":
            kw.update(p.get("tpu_llm_kwargs", {}))
        if "logits_processors" in kw:
            kw["logits_processors"] = [_resolve(x) for x in kw["logits_processors"]]
        self.llm = LLM(**kw)

        sampling = {"temperature": 0.0, "max_tokens": 8192, **p.get("sampling", {})}
        self.sp = SamplingParams(**sampling)
        self.prompt = PROMPTS.get(p.get("prompt"), p.get("prompt"))
        self.post = [POSTPROCESS[name] for name in p.get("postprocess", [])]

    def _message(self, img: Image.Image) -> list[dict]:
        p = self.params
        image_part = {"type": "image_pil", "image_pil": img}
        if self.prompt is None:
            content = [image_part]
        else:
            text_part = {"type": "text", "text": p.get("text_prefix", "") + self.prompt}
            content = [text_part, image_part] if p.get("text_first") else [image_part, text_part]
        return [{"role": "user", "content": content}]

    def predict(self, image_paths):
        p = self.params
        images = [_fit(Image.open(x).convert("RGB"), p.get("max_side"), p.get("max_pixels")) for x in image_paths]
        if p.get("mode") == "generate":   # raw prompt string (DeepSeek-OCR)
            outs = self.llm.generate([{"prompt": p["raw_prompt"], "multi_modal_data": {"image": im}} for im in images],
                                     self.sp, use_tqdm=False)
        else:
            outs = self.llm.chat([self._message(im) for im in images], self.sp, use_tqdm=False,
                                 **p.get("chat_kwargs", {}))
        preds = []
        for img, o in zip(images, outs):
            c = o.outputs[0]
            text = c.text
            for f in self.post:
                text = f(text, image=img)
            preds.append(Prediction(text=text.strip(), finish_reason=c.finish_reason, n_tokens=len(c.token_ids)))
        return preds

    def versions(self):
        return {k: pkg_version(k) for k in ("vllm", "vllm-tpu", "torch", "transformers")}

    def close(self):
        # Stop the EngineCore process; the runner also kills the process group as a backstop.
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
