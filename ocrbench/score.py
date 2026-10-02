"""Score a run directory: predictions + ground truth -> page_scores.csv and summary.csv.

Scoring needs no engine dependencies, so it can be re-run at any time against
saved predictions (e.g. after improving normalisation).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config
from .data.base import read_manifest
from .metrics import olmocr_tests
from .metrics.text import score_page

TEXT_TRACKS = ("books", "handwriting", "stress")


def hourly_usd(hw_tag: str, cu_override: float | None = None) -> float | None:
    costs = config.load_yaml("hardware_costs.yaml")
    cu = cu_override if cu_override is not None else (costs["runtimes"].get(hw_tag) or {}).get("cu_per_hour")
    return None if cu is None else cu * costs["usd_per_cu"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_manifest(job: dict, track: str, profile: str) -> Path | None:
    p = Path(job.get("tracks", {}).get(track, ""))
    if p.is_file():
        return p
    p = config.DATA_DIR / profile / track / "manifest.jsonl"   # results copied to another machine
    return p if p.is_file() else None


def score_engine_track(engine: str, track: str, preds: list[dict], manifest: Path) -> list[dict]:
    gt = {s.id: s for s in read_manifest(manifest)}
    rows = []
    for p in preds:
        s = gt.get(p["id"])
        if s is None:
            continue
        base = {"engine": engine, "track": track, "id": s.id, "seconds": p.get("seconds"),
                "n_tokens": p.get("n_tokens"), "page_error": bool(p.get("error"))}
        base.update({f"meta_{k}": v for k, v in s.meta.items() if k in ("book", "lang", "level", "layout", "category")})
        if track == "docs":
            results = olmocr_tests.evaluate(s.meta.get("tests", []), p["text"])
            base["tests_total"] = len(results)
            base["tests_passed"] = sum(r["passed"] for r in results)
            base["test_results"] = json.dumps(results)
            base["empty"] = not p["text"].strip()
        else:
            base.update(score_page(s.gt_text, p["text"], p.get("finish_reason")))
        rows.append(base)
    return rows


def _docs_pass_rate(df: pd.DataFrame) -> tuple[float | None, dict]:
    """olmOCR-bench style: pass rate per test category, then the mean across categories."""
    results = [r for js in df["test_results"].dropna() for r in json.loads(js)]
    if not results:
        return None, {}
    by_cat = pd.DataFrame(results).groupby("category")["passed"].mean()
    return float(by_cat.mean()), by_cat.round(4).to_dict()


def score_run(run_dir: Path, cu_override: float | None = None) -> pd.DataFrame:
    meta = json.loads((run_dir / "env.json").read_text())
    hw_tag, profile = meta["hardware"]["tag"], meta["profile"]["name"]
    if cu_override is None:
        cu_override = meta.get("cu_per_hour")   # set at run time, e.g. for TPU runtimes
    usd_h = hourly_usd(hw_tag, cu_override)
    page_rows, summary = [], []

    for status_path in sorted(run_dir.glob("*/status.json")):
        edir = status_path.parent
        status = json.loads(status_path.read_text())
        job = json.loads((edir / "job.json").read_text()) if (edir / "job.json").exists() else {}
        spec = job.get("spec", {})
        engine = status.get("engine", edir.name)
        common = {"run_id": meta["run_id"], "hardware": hw_tag, "engine": engine,
                  "engine_name": spec.get("name", engine), "family": spec.get("family"),
                  "kind": spec.get("kind"), "status": status.get("status"),
                  "error": status.get("error"), "load_s": status.get("load_s"),
                  "peak_vram_gb": status.get("peak_vram_gb"), "peak_rss_gb": status.get("peak_rss_gb"),
                  "size_b": spec.get("size_b"), "license": spec.get("license")}
        for track in meta["tracks"]:
            tstat = status.get("tracks", {}).get(track, {})
            manifest = _find_manifest(job, track, profile)
            preds = _read_jsonl(edir / track / "preds.jsonl")
            rows = score_engine_track(engine, track, preds, manifest) if (preds and manifest) else []
            page_rows += rows
            row = {**common, "track": track, "n": len(rows), "n_planned": tstat.get("n_planned"),
                   "partial": tstat.get("partial", False), "track_error": tstat.get("error"),
                   "pages_per_s": tstat.get("pages_per_s"), "latency_p50_s": tstat.get("latency_p50_s"),
                   "latency_p95_s": tstat.get("latency_p95_s")}
            if rows:
                df = pd.DataFrame(rows)
                row["empty_rate"] = df["empty"].mean()
                row["page_error_rate"] = df["page_error"].mean()
                if track == "docs":
                    row["docs_pass"], cats = _docs_pass_rate(df)
                    row["docs_categories"] = json.dumps(cats)
                    row["accuracy"] = row["docs_pass"]
                else:
                    for m in ("cer", "cer_dip", "wer"):
                        row[m] = df[m].clip(upper=1).mean()       # page-mean, capped so one runaway page can't dominate
                    row["cer_micro"] = df["cer_edits"].sum() / max(df["ref_chars"].sum(), 1)
                    for m in ("bow_recall", "bow_precision", "over_extraction"):
                        row[m] = df[m].mean()
                    row["loop_rate"] = df["loop"].mean()
                    row["accuracy"] = 1 - row["cer"]
            row["usd_per_hour"] = usd_h
            if usd_h and row.get("pages_per_s"):
                row["usd_per_1k_pages"] = usd_h / 3600 / row["pages_per_s"] * 1000
            summary.append(row)

    pages = pd.DataFrame(page_rows)
    if not pages.empty:
        pages.to_csv(run_dir / "page_scores.csv", index=False)
    out = pd.DataFrame(summary)
    out.to_csv(run_dir / "summary.csv", index=False)
    return out
