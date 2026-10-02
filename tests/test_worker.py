"""Run fake engines through the real runner -> worker subprocess path."""
import json
import time

import psutil

from ocrbench import runner
from ocrbench.data.base import Sample, write_manifest
from ocrbench.hardware import HardwareInfo


def _tracks(tmp_path):
    tracks = {}
    for t in ("books", "stress"):
        tracks[t] = write_manifest(tmp_path / t, [Sample(id=f"{t}{i}", track=t, image="x.png", gt_text="x")
                                                  for i in range(4)])
    return tracks


def _spec(impl):
    return {"id": impl.split(":")[1].lower(), "family": "classic", "impl": f"tests.fake_engines:{impl.split(':')[1]}"}


def test_dead_engine_stops_and_orphans_are_killed(tmp_path):
    hw = HardwareInfo(tag="cpu", kind="cpu")
    t = time.time()
    status = runner.run_engine(_spec("x:Dies"), hw, _tracks(tmp_path), tmp_path / "out", latency_n=0,
                               timeout_min=1, use_venvs=False)
    assert time.time() - t < 60, "runner waited on the orphaned child"
    assert status["status"] == "partial"
    assert "EngineDeadError" in status["error"]
    assert "stress" not in status["tracks"]          # stopped instead of failing every later page
    books = [json.loads(l) for l in open(tmp_path / "out/books/preds.jsonl")]
    assert len(books) == 2 and all(r["text"] for r in books)   # no empty "error" pages recorded
    orphans = [p for p in psutil.process_iter(["cmdline"])
               if (p.info["cmdline"] or [])[1:] == ["-c", "import time; time.sleep(300)"]]
    assert not orphans


def test_slow_load_does_not_eat_page_budget(tmp_path):
    hw = HardwareInfo(tag="cpu", kind="cpu")
    status = runner.run_engine(_spec("x:SlowLoad"), hw, _tracks(tmp_path), tmp_path / "out", latency_n=0,
                               timeout_min=2 / 60, use_venvs=False)
    assert status["status"] == "ok", status
    assert status["load_s"] >= 3
    assert status["tracks"]["books"]["n"] == 4
