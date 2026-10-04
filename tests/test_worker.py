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


def test_bad_page_in_latency_pass_costs_one_page_not_the_track(tmp_path):
    hw = HardwareInfo(tag="cpu", kind="cpu")
    samples = [Sample(id=f"b{i}", track="books", image="bad.png" if i == 1 else "x.png", gt_text="x")
               for i in range(4)]
    tracks = {"books": write_manifest(tmp_path / "books", samples)}
    status = runner.run_engine(_spec("x:BadPage"), hw, tracks, tmp_path / "out", latency_n=2,
                               timeout_min=1, use_venvs=False)
    assert status["status"] == "ok", status
    assert status["tracks"]["books"]["n"] == 4
    recs = [json.loads(l) for l in open(tmp_path / "out/books/preds.jsonl")]
    assert [bool(r.get("error")) for r in recs] == [False, True, False, False]


def test_slow_engine_gets_a_sample_of_every_track(tmp_path):
    hw = HardwareInfo(tag="cpu", kind="cpu")
    tracks = {t: write_manifest(tmp_path / t, [Sample(id=f"{t}{i}", track=t, image="x.png", gt_text="x")
                                               for i in range(5)])
              for t in ("books", "docs", "stress")}
    status = runner.run_engine(_spec("x:SlowPages"), hw, tracks, tmp_path / "out", latency_n=0,
                               timeout_min=1.8 / 60, use_venvs=False)
    assert status["status"] == "partial"
    n = {t: status["tracks"][t]["n"] for t in tracks}
    assert all(v > 0 for v in n.values()), n      # the last track isn't starved
    assert n["books"] < 5, n                       # the first track doesn't take the whole budget


def test_time_left_by_fast_tracks_goes_back_to_cut_tracks(tmp_path):
    hw = HardwareInfo(tag="cpu", kind="cpu")
    tracks = {"books": write_manifest(tmp_path / "books", [Sample(id=f"b{i}", track="books", image="x.png",
                                                                  gt_text="x") for i in range(8)]),
              "stress": write_manifest(tmp_path / "stress", [Sample(id=f"s{i}", track="stress", image="fast.png",
                                                                    gt_text="x") for i in range(8)])}
    # books needs 2.4 s but its even share is 1.35 s; stress is instant, so books resumes and finishes
    status = runner.run_engine(_spec("x:SlowPages"), hw, tracks, tmp_path / "out", latency_n=0,
                               timeout_min=2.7 / 60, use_venvs=False)
    assert status["tracks"]["books"]["n"] == 8, status["tracks"]
    assert status["status"] == "ok"
    ids = [json.loads(l)["id"] for l in open(tmp_path / "out/books/preds.jsonl")]
    assert ids == [f"b{i}" for i in range(8)]


def test_batches_shrink_to_fit_the_time_left(tmp_path):
    hw = HardwareInfo(tag="cpu", kind="cpu")
    tracks = {"books": write_manifest(tmp_path / "books", [Sample(id=f"b{i}", track="books", image="x.png",
                                                                  gt_text="x") for i in range(20)])}
    t = time.time()
    status = runner.run_engine(_spec("x:SlowBatches"), hw, tracks, tmp_path / "out", latency_n=2,
                               timeout_min=2.0 / 60, use_venvs=False)
    # one 32-page batch would take 6 s; sized batches stop near the 2 s budget instead
    assert status["status"] == "partial", status
    assert 3 <= status["tracks"]["books"]["n"] < 20
    assert status["wall_s"] < 6, status["wall_s"]


def test_cheap_repeat_pages_do_not_inflate_batch_size(tmp_path):
    hw = HardwareInfo(tag="cpu", kind="cpu")
    samples = [Sample(id=f"b{i}", track="books", image=f"p{i}.png", gt_text="x") for i in range(20)]
    tracks = {"books": write_manifest(tmp_path / "books", samples)}
    status = runner.run_engine(_spec("x:CachedRepeats"), hw, tracks, tmp_path / "out", latency_n=4,
                               timeout_min=3.0 / 60, use_venvs=False)
    # The first batch repeats the 4 latency pages, so it looks 3x faster than it is. Sizing the
    # next batch from that rate would start ~12 new pages (3.6 s) with ~1.2 s left.
    assert status["tracks"]["books"]["wall_s"] < 3, status["tracks"]["books"]
