"""Command line: python -m ocrbench <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _run_dir(arg: str) -> Path:
    if arg == "latest":
        runs = sorted(p.parent for p in config.RUNS_DIR.glob("*/env.json"))
        if not runs:
            sys.exit("no runs found under results/runs")
        return max(runs, key=lambda p: p.stat().st_mtime)
    p = Path(arg)
    return p if p.is_dir() else config.RUNS_DIR / arg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ocrbench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hardware", help="print detected hardware")

    p = sub.add_parser("prepare", help="download/render the datasets for a profile")
    p.add_argument("--profile", default="standard")
    p.add_argument("--tracks", help="comma list (default: all)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run", help="run the benchmark")
    p.add_argument("--profile", default="standard")
    p.add_argument("--engines", help="only these engine ids (comma list)")
    p.add_argument("--exclude", help="skip these engine ids")
    p.add_argument("--tracks", help="comma list (default: all)")
    p.add_argument("--run-id")
    p.add_argument("--timeout-min", type=float, help="override per-engine time budget")
    p.add_argument("--dry-run", action="store_true", help="show which engines would run, then stop")
    p.add_argument("--no-venv", action="store_true", help="run engines in the current interpreter (local dev)")
    p.add_argument("--cu-per-hour", type=float, help="Colab compute units/hour for this runtime (needed for TPU cost)")

    p = sub.add_parser("score", help="score a run directory")
    p.add_argument("run", nargs="?", default="latest")
    p.add_argument("--cu-per-hour", type=float, help="override Colab compute units/hour for cost")

    p = sub.add_parser("report", help="rebuild results/LEADERBOARD.md from all runs")
    p.add_argument("--rescore", action="store_true")

    p = sub.add_parser("publish", help="commit + push a run and the leaderboard (needs GITHUB_TOKEN)")
    p.add_argument("run", nargs="?", default="latest")
    p.add_argument("--branch", default="main")

    a = ap.parse_args(argv)
    if a.cmd == "hardware":
        from .hardware import detect

        print(json.dumps(detect().to_dict(), indent=2))
    elif a.cmd == "prepare":
        from .data import prepare_tracks

        for track, path in prepare_tracks(a.profile, _csv(a.tracks) or list(config.TRACKS), a.force).items():
            print(f"{track}: {path}")
    elif a.cmd == "run":
        from .runner import run

        run_dir = run(a.profile, _csv(a.tracks), _csv(a.engines), _csv(a.exclude), a.run_id,
                      a.dry_run, use_venvs=not a.no_venv, timeout_min=a.timeout_min, cu_per_hour=a.cu_per_hour)
        if run_dir:
            from .score import score_run

            s = score_run(run_dir)
            cols = [c for c in ("engine", "track", "status", "n", "accuracy", "pages_per_s") if c in s]
            print("\n" + s[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            print(f"\nrun saved to {run_dir}")
    elif a.cmd == "score":
        from .score import score_run

        print(score_run(_run_dir(a.run), a.cu_per_hour).to_string())
    elif a.cmd == "report":
        from .report import build

        print(f"wrote {build(rescore=a.rescore)}")
    elif a.cmd == "publish":
        from .publish import publish

        print(f"pushed {publish(_run_dir(a.run), a.branch)}")


if __name__ == "__main__":
    main()
