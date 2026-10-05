"""End-to-end scoring + leaderboard on fabricated run directories."""
import json

from ocrbench import report
from ocrbench.data.base import Sample, write_manifest
from ocrbench.score import score_run

GT = "The quick brown fox jumps over the lazy dog near the river bank."


def _make_data(root):
    tracks = {}
    for track in ("books", "handwriting", "stress"):
        samples = [Sample(id=f"{track}{i}", track=track, image=f"images/{i}.png", gt_text=GT,
                          meta={"level": lvl, "lang": "en"} if track == "stress" else {"lang": "en"})
                   for i, lvl in enumerate(["clean", "blur"])]
        tracks[track] = write_manifest(root / track, samples)
    tests = [{"id": "t1", "type": "present", "text": "quick brown fox", "max_diffs": 0, "category": "old_scans"}]
    tracks["docs"] = write_manifest(root / "docs", [Sample(id="d0", track="docs", image="images/d.png",
                                                            meta={"category": "old_scans", "tests": tests})])
    return tracks


def _make_run(runs, data, run_id, hw, engines):
    rd = runs / run_id
    rd.mkdir(parents=True)
    plan = [{"id": e, "family": "x", "run": True, "skip_reason": None} for e in engines]
    plan.append({"id": "big-model", "family": "vllm", "run": False, "skip_reason": "insufficient_vram: needs 30 GB"})
    (rd / "env.json").write_text(json.dumps({"run_id": run_id, "profile": {"name": "standard"},
                                             "tracks": list(data), "hardware": {"tag": hw}, "started": 1, "plan": plan}))
    for engine, (text, pps, status) in engines.items():
        ed = rd / engine
        ed.mkdir()
        (ed / "job.json").write_text(json.dumps({"spec": {"id": engine, "name": engine.title(), "kind": "classic"},
                                                 "tracks": {k: str(v) for k, v in data.items()}}))
        tracks_status = {}
        if status == "ok":
            for track, manifest in data.items():
                (ed / track).mkdir()
                ids = [json.loads(l)["id"] for l in open(manifest)]
                with open(ed / track / "preds.jsonl", "w") as f:
                    for i in ids:
                        f.write(json.dumps({"id": i, "text": text, "seconds": 1 / pps}) + "\n")
                tracks_status[track] = {"n": len(ids), "n_planned": len(ids), "pages_per_s": pps}
        (ed / "status.json").write_text(json.dumps({"engine": engine, "status": status, "tracks": tracks_status,
                                                    "error": None if status == "ok" else "CUDA out of memory"}))
    return rd


def test_leaderboard_end_to_end(tmp_path):
    data = _make_data(tmp_path / "data")
    runs = tmp_path / "results" / "runs"
    rd = _make_run(runs, data, "r1", "T4", {"good": (GT, 2.0, "ok"), "sloppy": ("The quick brwn fx jumps", 8.0, "ok"),
                                            "broken": ("", 1.0, "error")})
    _make_run(runs, data, "r2", "A100-40GB", {"good": (GT, 10.0, "ok")})

    summary = score_run(rd)
    good = summary[(summary.engine == "good") & (summary.track == "books")].iloc[0]
    assert good["cer"] == 0 and good["accuracy"] == 1
    assert summary[(summary.engine == "good") & (summary.track == "docs")].iloc[0]["docs_pass"] == 1.0
    # T4 at 1.19 CU/h * $0.0999 = $0.1189/h; 2 pages/s -> $0.0165 per 1k pages
    assert abs(good["usd_per_1k_pages"] - 1.19 * 0.0999 / 3600 / 2 * 1000) < 1e-9

    md = report.build(tmp_path / "results").read_text()
    lb = report.leaderboard(*report.load_runs(runs)[:1], "standard")
    assert lb.iloc[0]["engine"] == "good" and lb.iloc[0]["rank"] == 1
    assert lb.set_index("engine").loc["good", "best_pps_hw"] == "A100-40GB"
    assert "| Most accurate overall | Good |" in md
    assert "insufficient_vram" in md and "CUDA out of memory" in md
    assert (tmp_path / "results" / "report.html").exists()
    assert (tmp_path / "results" / "charts" / "pareto.png").exists()


def test_result_far_below_other_hardware_is_flagged_and_excluded(tmp_path):
    data = _make_data(tmp_path / "data")
    runs = tmp_path / "results" / "runs"
    _make_run(runs, data, "r1", "A100-40GB", {"good": (GT, 2.0, "ok")})
    _make_run(runs, data, "r2", "G4", {"good": ("額俞臺姓W逻裔0逮哥", 50.0, "ok")})   # garbage, but "fast"
    for rd in runs.iterdir():
        score_run(rd)
    summary = report.load_runs(runs)[0]
    sus = report.suspect_results(summary)
    assert set(sus["hardware"]) == {"G4"} and "books" in set(sus["track"])
    lb = report.leaderboard(summary, "standard").set_index("engine")
    assert lb.loc["good", "best_pps_hw"] == "A100-40GB"      # the garbage run's speed doesn't count
    md = report.build(tmp_path / "results").read_text()
    assert "suspect" in md and "likely a broken build" in md
