"""Commit a finished run plus the regenerated leaderboard, and push to GitHub.

Auth uses GITHUB_TOKEN passed as an HTTP header for each command, so the token is
never written to .git/config or printed. Several Colab runtimes may publish at
once: each run lives in its own directory (no conflicts), and the leaderboard is
regenerated after rebasing onto whatever was pushed in the meantime.
"""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

from . import config, report

LEADERBOARD_FILES = ["results/LEADERBOARD.md", "results/leaderboard.csv", "results/hardware.csv",
                     "results/report.html", "results/charts"]


def _git(*args: str, auth: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["git", "-C", str(config.REPO_ROOT)]
    if auth:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_TOKEN is not set; add it as a Colab secret")
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]
    res = subprocess.run(cmd + list(args), capture_output=True, text=True)
    if check and res.returncode != 0:
        # never echo the command line: it contains the auth header
        raise RuntimeError(f"git {args[0]} failed: {res.stderr.strip()[-600:]}")
    return res


def _ensure_identity() -> None:
    if not _git("config", "user.email", check=False).stdout.strip():
        _git("config", "user.email", "ocrbench@users.noreply.github.com")
        _git("config", "user.name", "ocrbench (Colab)")


def trim_logs(run_dir: Path, max_lines: int = 1500) -> None:
    """vLLM logs are long; keep only the tail so the repo stays small."""
    for log in run_dir.glob("*/worker.log"):
        lines = log.read_text(errors="replace").splitlines()
        if len(lines) > max_lines:
            log.write_text("\n".join([f"... [{len(lines) - max_lines} earlier lines trimmed]", *lines[-max_lines:]]) + "\n")


def _discard_generated() -> None:
    """Drop local copies of the generated leaderboard files. The notebook rebuilds them for
    display, and those edits would block the rebase; they are rebuilt after it anyway."""
    for f in LEADERBOARD_FILES:
        if _git("ls-files", "--error-unmatch", f, check=False).returncode == 0:
            _git("checkout", "HEAD", "--", f)
        _git("clean", "-fdq", "--", f)


def publish(run_dir: Path, branch: str = "main", retries: int = 4) -> str:
    from .score import score_run

    _ensure_identity()
    trim_logs(run_dir)
    if not (run_dir / "summary.csv").exists():
        score_run(run_dir)
    rel = run_dir.resolve().relative_to(config.REPO_ROOT)
    _git("add", str(rel))
    if _git("diff", "--cached", "--quiet", check=False).returncode != 0:   # not already committed
        _git("commit", "-m", f"Add benchmark run {run_dir.name}")

    for attempt in range(1, retries + 1):
        _discard_generated()
        _git("fetch", "origin", branch, auth=True)
        _git("rebase", "--autostash", f"origin/{branch}")
        report.build()
        _git("add", "-A", *[f for f in LEADERBOARD_FILES if (config.REPO_ROOT / f).exists()])
        if _git("diff", "--cached", "--quiet", check=False).returncode != 0:
            _git("commit", "-m", f"Update leaderboard after {run_dir.name}")
        push = _git("push", "origin", f"HEAD:{branch}", auth=True, check=False)
        if push.returncode == 0:
            return _git("rev-parse", "--short", "HEAD").stdout.strip()
        if not any(m in push.stderr for m in ("non-fast-forward", "fetch first", "[rejected]")):
            hint = ""
            if "403" in push.stderr or "not granted" in push.stderr:
                hint = ("\nGITHUB_TOKEN can read but not write this repo. Give the token "
                        "'Contents: Read and write' on the repository, then run publish again.")
            raise RuntimeError(f"git push failed: {push.stderr.strip()[-400:]}{hint}")
        print(f"push rejected (attempt {attempt}/{retries}); someone else pushed, retrying")
        # Drop only the regenerated-leaderboard commit; it is rebuilt on the next attempt.
        if "leaderboard" in _git("log", "-1", "--format=%s").stdout:
            _git("reset", "--hard", "HEAD~1")
    raise RuntimeError("could not push after several attempts: " + push.stderr.strip()[-300:])
