"""Aggregate every run under results/runs/ into the leaderboard (Markdown, CSV, charts, HTML).

Accuracy is hardware-independent, so each engine's accuracy comes from its best
run in the headline profile. Speed and cost are per hardware: one column per
Colab runtime you've run the notebook on.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import pandas as pd

from . import config
from .data.synthetic import LEVELS
from .score import hourly_usd, score_run

PROFILE_RANK = {"smoke": 1, "standard": 2, "full": 3}
TRACK_LABEL = {"books": "Books (historical)", "docs": "Docs (olmOCR-bench)",
               "handwriting": "Handwriting", "stress": "Stress (synthetic)"}
KIND_LABEL = {"classic": "Classic OCR", "ocr_vlm": "OCR VLM", "vlm": "General VLM"}
HW_ORDER = ["cpu", "T4", "L4", "A100-40GB", "A100-80GB", "G4", "H100", "TPU-v5e-1", "TPU-v6e-1"]

# Reference palette (dataviz skill): three categorical slots, validated all-pairs, plus blue ramp.
KIND_COLOR = {"classic": "#2a78d6", "ocr_vlm": "#eb6834", "vlm": "#1baf7a"}
KIND_MARKER = {"classic": "s", "ocr_vlm": "o", "vlm": "^"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ------------------------------------------------------------------ loading

def load_runs(runs_dir: Path = config.RUNS_DIR, rescore: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    summaries, pages, metas = [], [], []
    for env_path in sorted(runs_dir.glob("*/env.json")):
        run_dir = env_path.parent
        meta = json.loads(env_path.read_text())
        metas.append(meta)
        if rescore or not (run_dir / "summary.csv").exists():
            try:
                score_run(run_dir)
            except Exception as e:  # a half-written run shouldn't break the leaderboard
                print(f"warning: could not score {run_dir.name}: {e}")
                continue
        try:
            s = pd.read_csv(run_dir / "summary.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if s.empty:
            continue
        s["profile"] = meta["profile"]["name"]
        s["started"] = meta.get("started", 0)
        summaries.append(s)
        if (run_dir / "page_scores.csv").exists():
            p = pd.read_csv(run_dir / "page_scores.csv", low_memory=False)
            p["run_id"], p["profile"], p["hardware"] = meta["run_id"], meta["profile"]["name"], meta["hardware"]["tag"]
            pages.append(p)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    page_df = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
    return summary, page_df, metas


def _registry() -> dict[str, dict]:
    return {e["id"]: e for e in config.load_yaml("engines.yaml")["engines"]}


# ------------------------------------------------------------------ tables

def headline_profile(summary: pd.DataFrame) -> str | None:
    ok = summary[summary["n"] > 0] if not summary.empty else summary
    if ok.empty:
        return None
    return max(ok["profile"].unique(), key=lambda p: PROFILE_RANK.get(p, 0))


# An engine scoring this much lower on one runtime than on another is broken there (a build
# without kernels for that GPU), not slower or worse: PaddlePaddle's CUDA 12.6 build on a
# Blackwell G4 returned random characters, 0% vs 84% elsewhere.
SUSPECT_DROP = 0.25


def suspect_results(summary: pd.DataFrame) -> pd.DataFrame:
    """Rows (run x engine x track) whose accuracy is SUSPECT_DROP below the same engine's best
    accuracy on that track on other hardware, with that best for reference."""
    cols = ["run_id", "engine", "hardware", "track", "accuracy", "best", "best_hw"]
    if summary.empty:
        return pd.DataFrame(columns=cols)
    s = summary[(summary["n"] > 0) & summary["accuracy"].notna()]
    out = []
    for (_, _), g in s.groupby(["engine", "track"]):
        for _, r in g.iterrows():
            others = g[g["hardware"] != r["hardware"]]
            if others.empty:
                continue
            top = others.loc[others["accuracy"].idxmax()]
            if r["accuracy"] < top["accuracy"] - SUSPECT_DROP:
                out.append({**r[cols[:5]].to_dict(), "best": top["accuracy"], "best_hw": top["hardware"]})
    return pd.DataFrame(out, columns=cols)


def _trusted(summary: pd.DataFrame) -> pd.DataFrame:
    """Drop every track of an engine's run that has a suspect track: its speed is not real either."""
    bad = suspect_results(summary)[["run_id", "engine"]].drop_duplicates()
    if bad.empty:
        return summary
    keys = set(map(tuple, bad.values))
    return summary[[(r, e) not in keys for r, e in zip(summary["run_id"], summary["engine"])]]


def best_rows(summary: pd.DataFrame, profile: str) -> pd.DataFrame:
    """One row per engine x track: the run with the most scored pages, latest first."""
    summary = _trusted(summary)
    s = summary[(summary["profile"] == profile) & (summary["n"] > 0)]
    s = s.sort_values(["n", "started"], ascending=False)
    return s.drop_duplicates(["engine", "track"])


def throughput_by_hw(summary: pd.DataFrame) -> pd.DataFrame:
    """Overall pages/s per engine x hardware (total pages / total time across tracks), latest run."""
    rows = []
    summary = _trusted(summary)
    s = summary[(summary["n"] > 0) & summary["pages_per_s"].notna()]
    for (engine, hw, run_id), g in s.groupby(["engine", "hardware", "run_id"]):
        secs = (g["n"] / g["pages_per_s"]).sum()
        rows.append({"engine": engine, "hardware": hw, "run_id": run_id, "started": g["started"].iloc[0],
                     "profile_rank": PROFILE_RANK.get(g["profile"].iloc[0], 0),
                     "pages": g["n"].sum(), "pages_per_s": g["n"].sum() / secs if secs else None,
                     "latency_p50_s": g["latency_p50_s"].median(),
                     "peak_vram_gb": g["peak_vram_gb"].iloc[0], "load_s": g["load_s"].iloc[0],
                     "usd_per_hour": g["usd_per_hour"].iloc[0] if "usd_per_hour" in g else hourly_usd(hw)})
    if not rows:
        return pd.DataFrame(columns=["engine", "hardware", "pages_per_s", "usd_per_1k_pages"])
    df = pd.DataFrame(rows).sort_values(["profile_rank", "started"], ascending=False)
    df = df.drop_duplicates(["engine", "hardware"])
    df["usd_per_1k_pages"] = [
        (usd / 3600 / pps * 1000) if (pps and usd and not math.isnan(usd)) else None
        for usd, pps in zip(df["usd_per_hour"], df["pages_per_s"])]
    return df


def leaderboard(summary: pd.DataFrame, profile: str) -> pd.DataFrame:
    reg = _registry()
    best = best_rows(summary, profile)
    tput = throughput_by_hw(summary)
    rows = []
    for engine, g in best.groupby("engine"):
        spec = reg.get(engine, {})
        by_track = g.set_index("track")
        recorded = g.iloc[0]
        row = {"engine": engine, "name": spec.get("name") or recorded.get("engine_name") or engine,
               "kind": spec.get("kind") or recorded.get("kind") or "?",
               "size_b": spec.get("size_b"), "license": spec.get("license", "")}
        accs = []
        for track in config.TRACKS:
            if track in by_track.index:
                r = by_track.loc[track]
                row[f"{track}_acc"] = r["accuracy"]
                row[f"{track}_cer"] = r.get("cer")
                accs.append(r["accuracy"])
            else:
                row[f"{track}_acc"] = None
        complete = len(accs) == len(config.TRACKS) and all(pd.notna(a) for a in accs)
        row["overall"] = sum(accs) / len(accs) if complete else None
        text = g[g["track"] != "docs"]
        row["loop_rate"] = text["loop_rate"].mean() if "loop_rate" in text else None
        row["empty_rate"] = g["empty_rate"].mean() if "empty_rate" in g else None
        row["over_extraction"] = text["over_extraction"].mean() if "over_extraction" in text else None
        t = tput[tput["engine"] == engine]
        if not t.empty:
            fastest = t.loc[t["pages_per_s"].idxmax()]
            row["best_pps"], row["best_pps_hw"] = fastest["pages_per_s"], fastest["hardware"]
            priced = t.dropna(subset=["usd_per_1k_pages"])
            if not priced.empty:
                cheap = priced.loc[priced["usd_per_1k_pages"].idxmin()]
                row["min_usd_1k"], row["min_usd_hw"] = cheap["usd_per_1k_pages"], cheap["hardware"]
        row["cpu_ok"] = "cpu" in spec.get("requires", {}).get("hardware", [])
        rows.append(row)
    lb = pd.DataFrame(rows)
    if lb.empty:
        return lb
    lb = lb.sort_values("overall", ascending=False, na_position="last").reset_index(drop=True)
    lb.insert(0, "rank", [i + 1 if pd.notna(o) else None for i, o in enumerate(lb["overall"])])
    return lb


def robustness(pages: pd.DataFrame, profile: str) -> pd.DataFrame:
    if pages.empty or "meta_level" not in pages:
        return pd.DataFrame()
    p = pages[(pages["track"] == "stress") & (pages["profile"] == profile)]
    if p.empty:
        return pd.DataFrame()
    # latest run per engine
    latest = p.sort_values("run_id").groupby("engine")["run_id"].last()
    p = p[p["run_id"] == p["engine"].map(latest)]
    tab = p.assign(cer=p["cer"].clip(upper=1)).pivot_table(index="engine", columns="meta_level", values="cer", aggfunc="mean")
    tab = tab[[lv for lv in LEVELS if lv in tab.columns]]
    if "clean" in tab:
        degraded = [c for c in tab.columns if c != "clean"]
        tab["degradation"] = tab[degraded].mean(axis=1) - tab["clean"]
    return tab


def books_by_language(pages: pd.DataFrame, profile: str) -> pd.DataFrame:
    if pages.empty or "meta_lang" not in pages:
        return pd.DataFrame()
    p = pages[(pages["track"] == "books") & (pages["profile"] == profile)]
    if p.empty:
        return pd.DataFrame()
    latest = p.sort_values("run_id").groupby("engine")["run_id"].last()
    p = p[p["run_id"] == p["engine"].map(latest)]
    return p.assign(cer=p["cer"].clip(upper=1)).pivot_table(index="engine", columns="meta_lang", values="cer", aggfunc="mean")


def docs_categories(summary: pd.DataFrame, profile: str) -> pd.DataFrame:
    best = best_rows(summary, profile)
    d = best[(best["track"] == "docs") & best["docs_categories"].notna()] if "docs_categories" in best else pd.DataFrame()
    rows = {r["engine"]: json.loads(r["docs_categories"]) for _, r in d.iterrows()}
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def _tied_names(names: list[str], show: int = 3) -> str:
    if len(names) <= show:
        return ", ".join(names)
    return ", ".join(names[:show]) + f" (+{len(names) - show} more tied)"


def winners(lb: pd.DataFrame, rob: pd.DataFrame) -> list[tuple[str, str, str]]:
    out = []
    if lb.empty:
        return out

    def pick(df, col, label, fmt, largest=True, why="", tol=0.0005):
        d = df.dropna(subset=[col])
        if d.empty:
            return
        r = d.loc[d[col].idxmax() if largest else d[col].idxmin()]
        # Ties (within 0.05 points): name them all instead of picking one arbitrarily.
        tied = d[(d[col] - r[col]).abs() <= tol * max(1.0, abs(r[col]))]
        out.append((label, _tied_names([r["name"], *[n for n in tied["name"] if n != r["name"]]]), fmt(r) + why))

    pick(lb, "overall", "Most accurate overall", lambda r: f"{r['overall']:.1%} mean accuracy across the 4 tracks")
    for track in config.TRACKS:
        col = f"{track}_acc"
        unit = "tests passed" if track == "docs" else "char. accuracy (1 - CER)"
        pick(lb, col, f"Best on {TRACK_LABEL[track]}", lambda r, c=col, u=unit: f"{r[c]:.1%} {u}")
    pick(lb, "best_pps", "Fastest", lambda r: f"{r['best_pps']:.2f} pages/s on {r['best_pps_hw']}")
    if "min_usd_1k" in lb and lb["overall"].notna().any():
        top = lb["overall"].max()
        near = lb[(lb["overall"] >= top - 0.05)]
        pick(near, "min_usd_1k", "Best value", lambda r: f"${r['min_usd_1k']:.3f} per 1k pages on {r['min_usd_hw']} "
             f"at {r['overall']:.1%} accuracy", largest=False, why=" (cheapest within 5 points of the top score)")
        pick(lb, "min_usd_1k", "Cheapest at scale",
             lambda r: f"${r['min_usd_1k']:.3f} per 1k pages (${r['min_usd_1k'] * 1000:,.0f} per 1M) on {r['min_usd_hw']}",
             largest=False)
    pick(lb[lb["cpu_ok"]], "overall", "Best CPU-capable engine", lambda r: f"{r['overall']:.1%} mean accuracy, no GPU required")
    if not rob.empty and "degradation" in rob:
        usable = rob[rob["clean"] < 0.10] if "clean" in rob else rob
        if not usable.empty:
            best = usable["degradation"].min()
            names = lb.set_index("engine")["name"]
            tied = usable[usable["degradation"] <= best + 0.0005].sort_values("degradation").index
            out.append(("Most robust to bad scans", _tied_names([names.get(e, e) for e in tied]),
                        f"CER rises only {best:.1%} from clean to degraded on average" if best > 0 else
                        f"CER no higher on degraded scans than on clean ones ({best:+.1%} on average)"))
    rel = lb.dropna(subset=["loop_rate"]).copy()
    if not rel.empty:
        rel["fail"] = rel["loop_rate"].fillna(0) + rel["empty_rate"].fillna(0)
        best = rel["fail"].min()
        tied = rel[rel["fail"] <= best + 0.0005].sort_values("overall", ascending=False)
        out.append(("Most reliable (fewest loops/empty pages)", _tied_names(list(tied["name"])),
                    f"{best:.1%} of pages looped or came back empty"))
    return out


# ------------------------------------------------------------------ formatting

def _pct(v, digits=1):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.{digits}f}%"


def _num(v, fmt="{:.2f}"):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def _bold_best(values: list[str], raw: list, largest: bool) -> list[str]:
    nums = [x for x in raw if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not nums:
        return values
    best = max(nums) if largest else min(nums)
    return [f"**{v}**" if (r is not None and not (isinstance(r, float) and math.isnan(r)) and r == best) else v
            for v, r in zip(values, raw)]


def _md_table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def leaderboard_md(lb: pd.DataFrame) -> str:
    cols = {
        "books": [_pct(1 - a) if a is not None and pd.notna(a) else "–" for a in lb["books_acc"]],
        "docs": [_pct(a) for a in lb["docs_acc"]],
        "handwriting": [_pct(1 - a) if pd.notna(a) else "–" for a in lb["handwriting_acc"]],
        "stress": [_pct(1 - a) if pd.notna(a) else "–" for a in lb["stress_acc"]],
    }
    cols["books"] = _bold_best(cols["books"], list(lb["books_acc"]), True)
    cols["docs"] = _bold_best(cols["docs"], list(lb["docs_acc"]), True)
    cols["handwriting"] = _bold_best(cols["handwriting"], list(lb["handwriting_acc"]), True)
    cols["stress"] = _bold_best(cols["stress"], list(lb["stress_acc"]), True)
    overall = _bold_best([_pct(v) for v in lb["overall"]], list(lb["overall"]), True)
    pps = _bold_best([_num(v) + (f" ({h})" if isinstance(h, str) else "") for v, h in
                      zip(lb.get("best_pps", [None] * len(lb)), lb.get("best_pps_hw", [None] * len(lb)))],
                     list(lb.get("best_pps", [None] * len(lb))), True)
    usd = _bold_best([("$" + _num(v, "{:.3f}")) if pd.notna(v) else "–" for v in lb.get("min_usd_1k", [None] * len(lb))],
                     list(lb.get("min_usd_1k", [None] * len(lb))), False)
    rows = []
    for i, r in lb.iterrows():
        size = f"{r['size_b']:g}B" if pd.notna(r.get("size_b")) and r.get("size_b") else "–"
        rows.append([_num(r["rank"], "{:.0f}"), r["name"], KIND_LABEL.get(r["kind"], r["kind"]), size,
                     overall[i], cols["books"][i], cols["docs"][i], cols["handwriting"][i], cols["stress"][i],
                     pps[i], usd[i], _pct(r.get("loop_rate"))])
    return _md_table(["#", "Engine", "Type", "Params", "Overall ↑", "Books CER ↓", "Docs pass ↑",
                      "Handwriting CER ↓", "Stress CER ↓", "Best pages/s ↑", "Min $/1k pages ↓", "Loop rate ↓"],
                     rows, align="rllrrrrrrrrr")


def hardware_md(tput: pd.DataFrame, names: dict[str, str]) -> str:
    if tput.empty:
        return "_No timing data yet._"
    hws = [h for h in HW_ORDER if h in set(tput["hardware"])] + sorted(set(tput["hardware"]) - set(HW_ORDER))
    rows = []
    order = tput.groupby("engine")["pages_per_s"].max().sort_values(ascending=False).index
    for eng in order:
        t = tput[tput["engine"] == eng].set_index("hardware")
        cells = []
        for hw in hws:
            if hw in t.index:
                r = t.loc[hw]
                usd = f" · ${r['usd_per_1k_pages']:.3f}" if pd.notna(r["usd_per_1k_pages"]) else ""
                cells.append(f"{r['pages_per_s']:.2f}{usd}")
            else:
                cells.append("–")
        rows.append([names.get(eng, eng), *cells])
    return _md_table(["Engine", *hws], rows)


def heat_md(tab: pd.DataFrame, names: dict[str, str], cols: list[str] | None = None, pct=True) -> str:
    if tab.empty:
        return "_No data yet._"
    cols = cols or list(tab.columns)
    tab = tab.sort_values(cols[0]) if cols else tab
    rows = [[names.get(e, e), *[(_pct(tab.loc[e, c]) if pct else _num(tab.loc[e, c])) for c in cols]] for e in tab.index]
    return _md_table(["Engine", *cols], rows)


def issues_md(summary: pd.DataFrame, metas: list[dict], names: dict[str, str]) -> str:
    """Engines skipped on a runtime, or whose *latest* run there failed or was cut short.
    Failures fixed by a later rerun on the same hardware are not listed."""
    issues, seen = [], set()
    for m in sorted(metas, key=lambda m: m.get("started", 0), reverse=True):
        for p in m.get("plan", []):
            key = (m["hardware"]["tag"], p["id"])
            if (not p["run"] and key not in seen and p["skip_reason"] != "excluded"
                    and not str(p["skip_reason"]).startswith("not_default")):
                issues.append([key[0], names.get(p["id"], p["id"]), "skipped", p["skip_reason"]])
            seen.add(key)
    if not summary.empty:
        latest = summary.sort_values("started").groupby(["engine", "hardware"])["run_id"].last()
        for (engine, hw), run_id in latest.items():
            rows = summary[(summary["engine"] == engine) & (summary["hardware"] == hw) & (summary["run_id"] == run_id)]
            status = rows["status"].iloc[0]
            if status not in ("error", "partial"):
                continue
            err = next((str(x) for x in [*rows["error"], *rows.get("track_error", [])] if pd.notna(x) and x), None)
            if err is None:
                done, planned = rows["n"].sum(), rows["n_planned"].fillna(0).sum()
                err = f"time budget ran out: {done:.0f} of {planned:.0f} pages scored"
            issues.append([hw, names.get(engine, engine), status, err[:140]])
    sus = suspect_results(summary)
    if not sus.empty:
        latest_runs = set(latest.items()) if not summary.empty else set()
        for (engine, hw, run_id), g in sus.groupby(["engine", "hardware", "run_id"]):
            if ((engine, hw), run_id) not in latest_runs:
                continue
            what = ", ".join(f"{t} {a:.0%} vs {b:.0%} on {bh}"
                             for t, a, b, bh in zip(g["track"], g["accuracy"], g["best"], g["best_hw"]))
            issues.append([hw, names.get(engine, engine), "suspect",
                           f"accuracy far below other hardware ({what}); likely a broken build here, "
                           "results excluded"])
    return _md_table(["Hardware", "Engine", "Status", "Reason"], issues, align="llll") if issues else "_None._"


# ------------------------------------------------------------------ charts

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _place_labels(fig, ax, points) -> None:
    """Direct labels that try a few positions around each point to avoid overlapping."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    offsets = [(6, 4, "left"), (6, -11, "left"), (-6, 4, "right"), (-6, -11, "right"), (6, 14, "left"), (6, -21, "left")]
    for name, x, y in sorted(points, key=lambda p: -p[2]):
        for dx, dy, ha in offsets:
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, ha=ha)
            box = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(b) for b in placed):
                break
            t.remove()
        else:   # nowhere free: keep the default position
            t = ax.annotate(name, (x, y), xytext=offsets[0][:2], textcoords="offset points", fontsize=8, color=INK)
            box = t.get_window_extent(renderer)
        placed.append(box)


def pareto_chart(lb: pd.DataFrame, path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = lb.dropna(subset=["overall", "best_pps"]) if "best_pps" in lb else pd.DataFrame()
    if d.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    _style(ax)
    for kind, g in d.groupby("kind"):
        ax.scatter(g["best_pps"], g["overall"] * 100, s=70, color=KIND_COLOR.get(kind, INK_2),
                   marker=KIND_MARKER.get(kind, "o"), edgecolor=SURFACE, linewidth=2,
                   label=KIND_LABEL.get(kind, kind), zorder=3)
    _place_labels(fig, ax, [(r["name"], r["best_pps"], r["overall"] * 100) for _, r in d.iterrows()])
    # Pareto frontier: nothing else is both faster and more accurate.
    front, best = [], -1
    for _, r in d.sort_values("best_pps", ascending=False).iterrows():
        if r["overall"] > best:
            front.append(r)
            best = r["overall"]
    if len(front) > 1:
        f = pd.DataFrame(front).sort_values("best_pps")
        ax.plot(f["best_pps"], f["overall"] * 100, color=INK_2, linewidth=1.5, linestyle="--", zorder=2,
                label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Throughput, pages per second (log scale, fastest hardware tested)", color=INK_2)
    ax.set_ylabel("Overall accuracy, %", color=INK_2)
    ax.set_title("Accuracy vs. speed: up and to the right is better", color=INK, loc="left", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def heatmap_chart(tab: pd.DataFrame, names: dict[str, str], title: str, path: Path, fmt=lambda v: f"{v * 100:.0f}",
                  vmax: float | None = None) -> bool:
    """Sequential single-hue heatmap: darker = larger value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if tab.empty:
        return False
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    vals = tab.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.1 * tab.shape[1] + 3.2, 0.42 * tab.shape[0] + 1.6), facecolor=SURFACE)
    vmax = vmax or (float(pd.DataFrame(vals).max().max()) or 1)
    ax.imshow(vals, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), [str(c) for c in tab.columns], rotation=35, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(range(tab.shape[0]), [names.get(e, e) for e in tab.index], fontsize=9, color=INK)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = vals[i, j]
            if not math.isnan(v):
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8, color="#ffffff" if v / vmax > 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, tab.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, tab.shape[0])], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ main

def build(results_dir: Path = config.RESULTS_DIR, rescore: bool = False) -> Path:
    summary, pages, metas = load_runs(results_dir / "runs", rescore=rescore)
    out_md = results_dir / "LEADERBOARD.md"
    charts = results_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    profile = headline_profile(summary) if not summary.empty else None
    reg = _registry()
    if profile is None:
        names = {k: v.get("name", k) for k, v in reg.items()}
        out_md.write_text("# OCR Leaderboard\n\n_No results yet. Run `notebooks/ocr_benchmark_colab.ipynb`._\n"
                          + (f"\n## Skipped and failed engines\n\n{issues_md(summary, metas, names)}\n" if metas else ""))
        return out_md

    lb = leaderboard(summary, profile)
    names = {**{k: v.get("name", k) for k, v in reg.items()}, **dict(zip(lb["engine"], lb["name"]))}
    tput = throughput_by_hw(summary)
    rob = robustness(pages, profile)
    langs = books_by_language(pages, profile)
    cats = docs_categories(summary, profile)
    wins = winners(lb, rob)
    lb.to_csv(results_dir / "leaderboard.csv", index=False)
    tput.to_csv(results_dir / "hardware.csv", index=False)

    has_pareto = pareto_chart(lb, charts / "pareto.png")
    track_tab = lb.set_index("engine")[[f"{t}_acc" for t in config.TRACKS]].rename(
        columns={f"{t}_acc": TRACK_LABEL[t] for t in config.TRACKS}).dropna(how="all")
    has_tracks = heatmap_chart(track_tab, names, "Accuracy by material type (%, darker = better)",
                               charts / "tracks.png", vmax=1.0)
    rob_levels = rob.drop(columns=["degradation"], errors="ignore")
    has_rob = heatmap_chart(rob_levels.sort_values("clean") if "clean" in rob_levels else rob_levels, names,
                            "Stress test: CER by degradation (%, darker = worse)", charts / "robustness.png", vmax=1.0)

    runs_hw = sorted({m["hardware"]["tag"] for m in metas})
    n_pages = best_rows(summary, profile).groupby("track")["n"].max().to_dict()
    md = [
        "# OCR Leaderboard",
        "",
        f"Headline profile: **{profile}** · pages per track: "
        + ", ".join(f"{TRACK_LABEL[t]} {n_pages.get(t, 0)}" for t in config.TRACKS)
        + f" · hardware tested: {', '.join(runs_hw)} · runs: {len(metas)}",
        "",
        "Generated by `python -m ocrbench report`. Scoring is described in [README](../README.md#how-scoring-works).",
        "",
        "## Winners",
        "",
        _md_table(["Category", "Winner", "Why"], [list(w) for w in wins], align="lll"),
        "",
        "## Overall leaderboard",
        "",
        "Overall = mean of the four track accuracies (1 − CER for text tracks, test pass rate for docs). "
        "CER is in *reading* mode (ligatures, long-s and quote styles normalised; Markdown/HTML stripped). "
        "Speed and cost are for the fastest/cheapest Colab runtime tested; see the hardware table below.",
        "",
        leaderboard_md(lb),
        "",
    ]
    if has_pareto:
        md += ["![Accuracy vs speed](charts/pareto.png)", ""]
    if has_tracks:
        md += ["![Accuracy by track](charts/tracks.png)", ""]
    md += [
        "## Speed and cost by hardware",
        "",
        "Cells: pages/s · $ per 1,000 pages (Colab compute-unit pricing from `configs/hardware_costs.yaml`; "
        "TPU rates are unknown until you fill them in). Multiply $/1k by 1,000 for the cost of a 1M-page collection.",
        "",
        hardware_md(tput, names),
        "",
        "## Robustness ladder (synthetic stress track, CER)",
        "",
        "Same pages, progressively degraded. `degradation` = mean CER over degraded levels minus clean CER.",
        "",
        heat_md(rob, names, cols=list(rob.columns)),
        "",
    ]
    if has_rob:
        md += ["![Robustness](charts/robustness.png)", ""]
    md += [
        "## Historical books by language (CER)",
        "",
        heat_md(langs, names),
        "",
        "## Modern documents by olmOCR-bench category (pass rate)",
        "",
        heat_md(cats, names),
        "",
        "## Reliability",
        "",
        _md_table(["Engine", "Loop rate", "Empty pages", "Over-extraction"],
                  [[r["name"], _pct(r.get("loop_rate")), _pct(r.get("empty_rate")), _pct(r.get("over_extraction"))]
                   for _, r in lb.iterrows()]),
        "",
        "## Skipped and failed engines",
        "",
    ]
    md.append(issues_md(summary, metas, names))
    text = "\n".join(md) + "\n"
    # Escape $ so GitHub/Colab don't render "$0.16 ... $36" as a math formula.
    out_md.write_text(text.replace("$", "\\$"))
    _html(results_dir, text)
    return out_md


def _html(results_dir: Path, md_text: str) -> None:
    """Self-contained HTML version (charts inlined) for sharing outside GitHub."""
    import re

    def img(m):
        p = results_dir / m.group(2)
        if not p.exists():
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img alt="{m.group(1)}" src="data:image/png;base64,{b64}">'

    body = []
    in_table = False
    for line in md_text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set(":-") for c in cells):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                body.append("<table>")
                in_table = True
            body.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("# "):
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("!["):
            body.append(re.sub(r"!\[([^\]]*)\]\(([^)]*)\)", img, line))
        elif line.strip():
            body.append(f"<p>{line}</p>")
    if in_table:
        body.append("</table>")
    html = "\n".join(body)
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)
    css = (":root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df}"
           "@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#383835}}"
           "body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px}"
           "table{border-collapse:collapse;display:block;overflow-x:auto;margin:12px 0;font-variant-numeric:tabular-nums}"
           "td,th{padding:4px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}"
           "td:first-child,th:first-child,td:nth-child(2){text-align:left}th{color:var(--ink2);font-weight:600}"
           "img{max-width:100%;border-radius:6px}p{color:var(--ink2)}")
    (results_dir / "report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>OCR Leaderboard</title><style>{css}</style></head><body>{html}</body></html>")
