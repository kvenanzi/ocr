"""Engine wiring tests. vLLM is mocked: we check that each registry entry produces the
request its model card prescribes, without needing a GPU."""
import importlib
import sys
import types

import pytest
from PIL import Image

from ocrbench import config
from ocrbench.engines import vllm_engine
from ocrbench.engines.prompts import PROMPTS
from ocrbench.runner import gate, load_registry
from ocrbench.hardware import HardwareInfo

REGISTRY = {e["id"]: e for e in load_registry()}


class FakeOutput:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text, finish_reason="stop", token_ids=[1, 2, 3])]


class FakeLLM:
    last = None

    def __init__(self, **kw):
        self.kw = kw
        self.calls = []
        FakeLLM.last = self

    def chat(self, messages, sp, use_tqdm=False, **kw):
        self.calls.append(("chat", messages, kw))
        return [FakeOutput("---\nprimary_language: en\n---\nHello <think>x</think>world") for _ in messages]

    def generate(self, prompts, sp, use_tqdm=False):
        self.calls.append(("generate", prompts, {}))
        return [FakeOutput("<|ref|>title<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n# Hi<｜end▁of▁sentence｜>") for _ in prompts]


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    mod.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod


@pytest.fixture
def page(tmp_path):
    p = tmp_path / "page.png"
    Image.new("RGB", (3000, 4000), "white").save(p)
    return str(p)


A100 = {"kind": "gpu", "bf16": True}
T4 = {"kind": "gpu", "bf16": False}


def _engine(eid, hw=A100):
    e = vllm_engine.ENGINE(REGISTRY[eid], hw)
    e.load()
    return e


def test_registry_is_consistent():
    ids = [e["id"] for e in load_registry()]
    assert len(ids) == len(set(ids))
    for e in load_registry():
        for key in ("id", "name", "kind", "family", "impl", "license", "requires"):
            assert key in e, (e["id"], key)
        assert e["kind"] in ("classic", "ocr_vlm", "vlm")
        importlib.import_module(f"ocrbench.engines.{e['impl'].split(':')[0]}")
        p = e.get("params", {})
        for name in p.get("postprocess", []):
            assert name in vllm_engine.POSTPROCESS, (e["id"], name)
        if e["impl"] == "vllm_engine":
            assert "model" in p
            assert p.get("mode") == "generate" or "prompt" in p, e["id"]


def test_olmocr_text_first_and_front_matter(fake_vllm, page):
    e = _engine("olmocr2")
    pred = e.predict([page])[0]
    kind, msgs, _ = FakeLLM.last.calls[-1]
    content = msgs[0][0]["content"]
    assert [c["type"] for c in content] == ["text", "image_pil"]
    assert content[0]["text"] == PROMPTS["OLMOCR_V4"]
    assert max(content[1]["image_pil"].size) == 1288          # resized per card
    assert pred.text == "Hello <think>x</think>world"         # only the front matter is configured to be stripped
    assert pred.finish_reason == "stop" and pred.n_tokens == 3


def test_dots_prefix_and_content_format(fake_vllm, page):
    e = _engine("dots-mocr")
    e.predict([page])
    _, msgs, kw = FakeLLM.last.calls[-1]
    content = msgs[0][0]["content"]
    assert [c["type"] for c in content] == ["image_pil", "text"]
    assert content[1]["text"].startswith("<|img|><|imgpad|><|endofimg|>Extract the text")
    assert kw == {"chat_template_content_format": "string"}
    assert FakeLLM.last.kw["trust_remote_code"] is True


def test_lighton_image_only(fake_vllm, page):
    e = _engine("lightonocr2")
    e.predict([page])
    content = FakeLLM.last.calls[-1][1][0][0]["content"]
    assert [c["type"] for c in content] == ["image_pil"]


def test_deepseek_raw_generate(fake_vllm, page, monkeypatch):
    fake_lp = types.ModuleType("vllm.model_executor.models.deepseek_ocr")
    fake_lp.NGramPerReqLogitsProcessor = object
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.deepseek_ocr", fake_lp)
    e = _engine("deepseek-ocr2")
    pred = e.predict([page])[0]
    kind, prompts, _ = FakeLLM.last.calls[-1]
    assert kind == "generate"
    assert prompts[0]["prompt"].startswith("<image>\n<|grounding|>")
    assert FakeLLM.last.kw["logits_processors"] == [object]
    assert e.sp["skip_special_tokens"] is False
    assert pred.text == "# Hi"


def test_t4_dtype(fake_vllm):
    assert _engine("paddleocr-vl", T4).llm.kw["dtype"] == "float16"
    assert _engine("granite-docling", T4).llm.kw["dtype"] == "float32"
    assert "dtype" not in _engine("paddleocr-vl", A100).llm.kw


def test_thinking_disabled_for_qwen35(fake_vllm, page):
    e = _engine("qwen3.5-9b")
    e.predict([page])
    assert FakeLLM.last.calls[-1][2] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert e.params["model"] == "Qwen/Qwen3.5-9B"            # YAML merge kept the override


def test_trim_repeats():
    text = "intro " * 2000 + "abc " * 300
    out = vllm_engine._trim_repeats(text)
    assert len(out) < len(text) and out.startswith("intro")


def test_gating():
    t4 = HardwareInfo(tag="T4", kind="gpu", vram_gb=15.0, compute_capability=7.5)
    p100 = HardwareInfo(tag="P100", kind="gpu", vram_gb=16.0, compute_capability=6.0)
    v5e = HardwareInfo(tag="TPU-v5e-1", kind="tpu", vram_gb=16)
    assert gate(REGISTRY["gemma4-e4b"], t4).startswith("insufficient_vram")
    assert gate(REGISTRY["paddleocr-vl"], p100).startswith("gpu_too_old")
    assert gate(REGISTRY["qwen2.5-vl-3b"], v5e) is None
    assert gate(REGISTRY["qwen2.5-vl-7b"], v5e).startswith("insufficient_hbm")
    assert gate(REGISTRY["dots-mocr"], v5e).startswith("unsupported_hardware")
    assert gate(REGISTRY["tesseract"], v5e) is None


def test_prefix_caching_off(fake_vllm, page):
    """The batch pass re-reads the latency pass's pages; cache hits would inflate throughput."""
    _engine("glm-ocr").predict([page])
    assert FakeLLM.last.kw["enable_prefix_caching"] is False


def test_paddle_build_follows_gpu_generation(monkeypatch):
    from ocrbench import envs
    from ocrbench.hardware import HardwareInfo

    def steps(cc):
        monkeypatch.setattr(envs, "detect", lambda: HardwareInfo(tag="x", kind="gpu", compute_capability=cc))
        return " ".join(envs._recipe("paddle")["steps"][0])
    assert "/cu126/" in steps(7.5) and "/cu126/" in steps(8.0)
    assert "/cu129/" in steps(12.0)     # G4, RTX PRO 6000 Blackwell


def test_tpu_plan_runs_cpu_engines_and_qwen25():
    from ocrbench.hardware import HardwareInfo
    from ocrbench.runner import plan
    rows = {r["id"]: r for r in plan(HardwareInfo(tag="TPU-v6e-1", kind="tpu", vram_gb=32.0))}
    for eid in ("tesseract", "ocrmypdf", "rapidocr", "easyocr", "doctr", "qwen2.5-vl-3b", "qwen2.5-vl-7b"):
        assert rows[eid]["run"], (eid, rows[eid]["skip_reason"])
    assert rows["qwen2.5-vl-7b"]["family"] == "vllm-tpu"


def test_vllm_tpu_env_uses_python_312():
    from ocrbench import envs
    assert envs.RECIPES["vllm-tpu"]["python"] == "3.12"
