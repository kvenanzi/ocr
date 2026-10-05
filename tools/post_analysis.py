#!/usr/bin/env python3
"""Every number and figure in docs/post/README.md, computed from results/runs/.

    python tools/post_analysis.py            # writes docs/post/figures/*.png and data.json

Accuracy comes from one reference runtime, the G4, the only one on which all 22 engines
scored every page with the final code (PaddleOCR from its rerun after the CUDA 12.9 fix).
Speed and cost come from every runtime. Intervals are percentile bootstraps over pages.
The FineBooks ground truth is downloaded (metadata only) to measure the characters in it
that no engine can produce.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocrbench import config, report  # noqa: E402
from ocrbench.data.synthetic import LEVELS  # noqa: E402
from ocrbench.metrics.normalize import reading  # noqa: E402
from ocrbench.report import BLUES, GRID, INK, INK_2, KIND_COLOR, KIND_LABEL, KIND_MARKER, SURFACE  # noqa: E402

OUT = config.REPO_ROOT / "docs" / "post" / "figures"
REF_RUN = "20261004-163611_G4_standard"
REF_PADDLE = "20261004-193333_G4_standard"     # PaddleOCR rerun with the CUDA 12.9 build
TRACKS = ["books", "docs", "handwriting", "stress"]
PAGE_TRACKS = ["books", "docs", "stress"]       # full pages; handwriting samples are single lines
HW = ["T4", "L4", "A100-40GB", "G4", "TPU-v6e-1"]
HW_COLOR = dict(zip(HW, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))   # validated, slots 1-5
HW_MARKER = dict(zip(HW, ["o", "s", "^", "D", "v"]))
N_BOOT = 2000
RNG = np.random.default_rng(0)


def reference(df: pd.DataFrame) -> pd.DataFrame:
    return df[((df.run_id == REF_RUN) & (df.engine != "paddleocr")) | (df.run_id == REF_PADDLE)]


# ------------------------------------------------------------------ accuracy with intervals

def _text_acc(cer: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return 1 - np.minimum(cer, 1)[idx].mean(axis=-1)


def _docs_acc(tests: list[list[dict]], idx: np.ndarray) -> np.ndarray:
    """olmOCR-bench score: pass rate per category, then the mean over categories."""
    cats = sorted({t["category"] for page in tests for t in page})
    # per page: passed and total per category
    passed = np.array([[sum(t["passed"] for t in page if t["category"] == c) for c in cats] for page in tests], float)
    total = np.array([[sum(1 for t in page if t["category"] == c) for c in cats] for page in tests], float)
    p, n = passed[idx].sum(axis=-2), total[idx].sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(p / n, axis=-1)


def accuracy_table(pages: pd.DataFrame) -> pd.DataFrame:
    rows, boots = [], {}
    for (engine, track), g in pages.groupby(["engine", "track"]):
        g = g.sort_values("id")
        n = len(g)
        idx = RNG.integers(0, n, size=(N_BOOT, n))
        full = np.arange(n)[None, :]
        if track == "docs":
            tests = [json.loads(t) if isinstance(t, str) else [] for t in g["test_results"]]
            point, boot = _docs_acc(tests, full)[0], _docs_acc(tests, idx)
        else:
            cer = g["cer"].to_numpy(float)
            point, boot = _text_acc(cer, full)[0], _text_acc(cer, idx)
        boots[(engine, track)] = boot
        rows.append({"engine": engine, "track": track, "n": n, "acc": point,
                     "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    acc = pd.DataFrame(rows)
    for engine in acc.engine.unique():
        b = np.mean([boots[(engine, t)] for t in TRACKS], axis=0)
        point = acc[acc.engine == engine].set_index("track").loc[TRACKS, "acc"].mean()
        acc.loc[len(acc)] = {"engine": engine, "track": "overall", "n": None, "acc": point,
                             "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)}
    return acc


# ------------------------------------------------------------------ speed and cost

def speed_table(summary: pd.DataFrame, metas: list[dict]) -> pd.DataFrame:
    """Pages/s over the full-page tracks, and lines/s on handwriting, per engine x runtime.
    For each pair: the trusted standard run with the most full pages scored, then the latest."""
    s = report._trusted(summary[summary.profile == "standard"])
    s = s[(s.n > 0) & s.pages_per_s.notna()]
    rows = []
    for (engine, hw, run_id), g in s.groupby(["engine", "hardware", "run_id"]):
        pg = g[g.track.isin(PAGE_TRACKS)]
        secs = (pg.n / pg.pages_per_s).sum()
        hwr = g[g.track == "handwriting"]
        usd_h = g.usd_per_hour.iloc[0]
        pps = pg.n.sum() / secs if secs else None
        rows.append({"engine": engine, "hardware": hw, "run_id": run_id, "started": g.started.iloc[0],
                     "pages": int(pg.n.sum()), "planned": int(pg.n_planned.sum()), "pages_per_s": pps,
                     "lines_per_s": float(hwr.pages_per_s.iloc[0]) if len(hwr) else None,
                     "latency_p50_s": float(pg.latency_p50_s.median()) if len(pg) else None,
                     "usd_per_hour": usd_h,
                     "usd_per_1k": (usd_h / 3600 / pps * 1000) if pps and usd_h and not math.isnan(usd_h) else None,
                     "peak_vram_gb": g.peak_vram_gb.iloc[0], "load_s": g.load_s.iloc[0]})
    df = pd.DataFrame(rows).sort_values(["pages", "started"], ascending=False)
    return df.drop_duplicates(["engine", "hardware"]).sort_values(["engine", "hardware"])


# ------------------------------------------------------------------ figures

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "regular"})
    return plt


def fig_accuracy_cost(acc: pd.DataFrame, speed: pd.DataFrame, names: dict, kinds: dict, path: Path) -> list[str]:
    plt = _mpl()
    ov = acc[acc.track == "overall"].set_index("engine")
    cheapest = speed.dropna(subset=["usd_per_1k"]).sort_values("usd_per_1k").drop_duplicates("engine").set_index("engine")
    d = ov.join(cheapest[["usd_per_1k", "hardware"]], how="inner")
    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor=SURFACE)
    report._style(ax)
    for kind in ["classic", "ocr_vlm", "vlm"]:
        g = d[[kinds[e] == kind for e in d.index]]
        ax.errorbar(g.usd_per_1k, g.acc * 100, yerr=[(g.acc - g.lo) * 100, (g.hi - g.acc) * 100], fmt="none",
                    ecolor=KIND_COLOR[kind], elinewidth=1, alpha=0.6, zorder=2)
        ax.scatter(g.usd_per_1k, g.acc * 100, s=70, color=KIND_COLOR[kind], marker=KIND_MARKER[kind],
                   edgecolor=SURFACE, linewidth=2, label=KIND_LABEL[kind], zorder=3)
    # Pareto set: no other engine is both cheaper and more accurate.
    front, best = [], -1.0
    for e, r in d.sort_values("usd_per_1k").iterrows():
        if r.acc > best:
            front.append(e)
            best = r.acc
    f = d.loc[front].sort_values("usd_per_1k")
    ax.plot(f.usd_per_1k, f.acc * 100, color=INK_2, linewidth=1.2, linestyle="--", zorder=1, label="Pareto set")
    report._place_labels(fig, ax, [(names[e], r.usd_per_1k, r.acc * 100) for e, r in d.iterrows()])
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.04, 0.06, 0.1, 0.15, 0.2, 0.3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.set_xlabel("Cost per 1,000 pages, US$ (log scale; cheapest runtime tested)", color=INK_2)
    ax.set_ylabel("Mean accuracy over the four tracks, % (95% CI)", color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return front


def fig_heat(tab: pd.DataFrame, names: dict, path: Path, cbar_label: str, fmt, vmax: float, figsize) -> None:
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    vals = tab.values.astype(float)
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    im = ax.imshow(np.minimum(vals, vmax), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), list(tab.columns), rotation=30, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(range(tab.shape[0]), [names.get(e, e) for e in tab.index], fontsize=9, color=INK)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = vals[i, j]
            if not math.isnan(v):
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                        color="#ffffff" if min(v, vmax) / vmax > 0.55 else INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, tab.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, tab.shape[0])], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, color=INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_hardware(speed: pd.DataFrame, order: list[str], names: dict, path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(order) + 1.6), facecolor=SURFACE)
    report._style(ax)
    ax.grid(True, axis="y", color=SURFACE)
    y = {e: i for i, e in enumerate(order)}
    for hw in HW:
        g = speed[(speed.hardware == hw) & speed.engine.isin(order)].dropna(subset=["pages_per_s"])
        ax.scatter(g.pages_per_s, [y[e] for e in g.engine], s=46, color=HW_COLOR[hw], marker=HW_MARKER[hw],
                   edgecolor=SURFACE, linewidth=1.5, label=hw, zorder=3)
    ax.set_yticks(range(len(order)), [names[e] for e in order], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3, 1, 3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Full pages per second (books, docs, synthetic; log scale)", color=INK_2)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, pages, metas = report.load_runs(config.RUNS_DIR)
    reg = report._registry()
    names = {e: reg[e]["name"] for e in reg}
    kinds = {e: reg[e]["kind"] for e in reg}
    data: dict = {"reference_runs": [REF_RUN, REF_PADDLE], "n_boot": N_BOOT}

    ref_pages = reference(pages)
    acc = accuracy_table(ref_pages)
    wide = acc.pivot(index="engine", columns="track", values="acc")[TRACKS + ["overall"]].sort_values(
        "overall", ascending=False)
    order = wide.index.tolist()
    data["accuracy"] = {e: {t: {k: round(float(v), 4) for k, v in
                                acc[(acc.engine == e) & (acc.track == t)][["acc", "lo", "hi"]].iloc[0].items()}
                            for t in TRACKS + ["overall"]} for e in order}

    # Material dependence: rank agreement between tracks over the 22 engines, and within the VLMs.
    data["spearman"] = {}
    for label, sub in [("all", wide), ("vlm", wide[[kinds[e] != "classic" for e in wide.index]])]:
        corr = sub[TRACKS].corr(method="spearman")
        data["spearman"][label] = {f"{a}~{b}": round(float(corr.loc[a, b]), 3)
                                   for i, a in enumerate(TRACKS) for b in TRACKS[i + 1:]}
    data["track_winner"] = {t: names[wide[t].idxmax()] for t in TRACKS}

    # Accuracy agreement across runtimes (complete, trusted tracks only).
    s = report._trusted(summary[summary.profile == "standard"])
    s = s[(s.n == s.n_planned) & (s.n > 0)]
    spread = s.groupby(["engine", "track"]).accuracy.agg(["min", "max", "count"])
    spread = spread[spread["count"] > 1]
    worst = (spread["max"] - spread["min"]).sort_values(ascending=False)
    data["cross_hw_spread"] = {"max": round(float(worst.iloc[0]), 4), "at": list(worst.index[0]),
                               "median": round(float(worst.median()), 4)}

    # Books: by language, and the share of ground-truth characters no engine can produce.
    b = ref_pages[ref_pages.track == "books"].assign(c=lambda d: d.cer.clip(upper=1))
    by_lang = b.pivot_table(index="engine", columns="meta_lang", values="c", aggfunc="mean")
    data["books_by_lang_cer"] = by_lang.round(4).to_dict(orient="index")
    data["books_pages_per_book"] = b.groupby("meta_book").id.nunique().to_dict()
    try:
        from huggingface_hub import hf_hub_download
        gt = pd.read_parquet(hf_hub_download("finebooks/bhl-impact-gt", "metadata.parquet", repo_type="dataset"),
                             columns=["file_name", "BarCode", "text"])
        ids = {i.rsplit("_full", 1)[0] for i in b.id}
        gt = gt[gt.file_name.map(lambda f: Path(f).stem.rsplit("_full", 1)[0] in ids)]
        unmatchable = lambda t: sum(1 for ch in t if 0xE000 <= ord(ch) <= 0xF8FF or ch == "�")  # noqa: E731
        gt = gt.assign(norm=gt.text.map(reading))
        per_book = gt.groupby("BarCode").apply(lambda g: g.norm.map(unmatchable).sum() / g.norm.str.len().sum())
        data["books_unmatchable_char_share"] = per_book.round(4).to_dict()
        data["finebooks_total"] = {"pages": 2165, "books": 6}
    except Exception as e:  # network optional
        data["books_unmatchable_char_share"] = f"not computed: {e}"
    # Two-column book: classic engines that read across columns.
    data["conchologia_cer"] = b[b.meta_book == "conchologiaiconi05reev"].groupby("engine").c.mean().round(3).to_dict()

    # Docs: pass rate per olmOCR-bench category.
    d = reference(summary)
    d = d[d.track == "docs"]
    cats = pd.DataFrame({e: json.loads(c) if isinstance(c, str) else c for e, c in zip(d.engine, d.docs_categories)}).T
    data["docs_categories"] = cats.loc[order].round(4).to_dict(orient="index")

    # Handwriting: empty and capped pages, output length.
    h = ref_pages[ref_pages.track == "handwriting"]
    data["handwriting"] = {e: {"empty": int(g["empty"].sum()), "cer_ge_1": int((g.cer >= 1).sum()),
                               "loops": int(g["loop"].sum()), "len_ratio_median": round(float(g.len_ratio.median()), 3),
                               "tokens_median": float(g.n_tokens.median()) if g.n_tokens.notna().any() else None}
                           for e, g in h.groupby("engine")}

    # Synthetic pages: CER by degradation level, and engines with no errors anywhere.
    st = ref_pages[ref_pages.track == "stress"].assign(c=lambda d: d.cer.clip(upper=1))
    levels = list(LEVELS)
    lv = st.pivot_table(index="engine", columns="meta_level", values="c", aggfunc="mean")[levels]
    data["stress_by_level_cer"] = lv.round(4).to_dict(orient="index")
    data["stress_perfect"] = [names[e] for e, g in st.groupby("engine") if (g.cer == 0).all()]
    data["stress_by_layout_cer"] = st.pivot_table(index="engine", columns="meta_layout", values="c",
                                                  aggfunc="mean").round(4).to_dict(orient="index")

    # Speed and cost.
    speed = speed_table(summary, metas)
    data["speed"] = speed.drop(columns=["started"]).round(4).to_dict(orient="records")
    data["cpus"] = {m["hardware"]["tag"]: m["hardware"].get("cpu_count") for m in metas}
    data["cu_per_hour"] = {m["hardware"]["tag"]: m.get("cu_per_hour") for m in metas if m.get("cu_per_hour")}

    # TPU against G4 for the same models: single-page latency by track.
    tpu = {}
    for e in ["qwen2.5-vl-3b", "qwen2.5-vl-7b"]:
        for run in ["20261005-140416_TPU-v6e-1_standard", REF_RUN]:
            for t in ["books", "handwriting", "stress"]:
                tm = json.loads((config.RUNS_DIR / run / e / t / "timing.json").read_text())
                tpu[f"{e}|{run.split('_')[1]}|{t}"] = [round(x, 2) for x in tm.get("latencies_s", [])]
    data["tpu_vs_g4_latency"] = tpu

    # Figures.
    front = fig_accuracy_cost(acc, speed, names, kinds, OUT / "accuracy_cost.png")
    data["pareto_set"] = [names[e] for e in front]
    tab = wide[TRACKS].rename(columns={"books": "Books", "docs": "Docs", "handwriting": "Handwriting",
                                       "stress": "Synthetic"}) * 100
    fig_heat(tab, names, OUT / "tracks.png", "accuracy, %", lambda v: f"{v:.0f}", 100, (7.2, 8.6))
    lvt = lv.loc[order] * 100
    fig_heat(lvt, names, OUT / "degradation.png", "character error rate, % (capped at 30)", lambda v: f"{v:.0f}",
             30, (10.5, 8.6))
    speed_order = speed[speed.hardware == "G4"].sort_values("pages_per_s", ascending=False).engine.tolist()
    fig_hardware(speed, speed_order, names, OUT / "hardware.png")

    (OUT / "data.json").write_text(json.dumps(data, indent=1, default=str))
    pd.set_option("display.width", 250)
    print((wide * 100).round(1).to_string())
    print("pareto:", data["pareto_set"])
    print("spearman:", data["spearman"])
    print("spread:", data["cross_hw_spread"])
    print("unmatchable:", data["books_unmatchable_char_share"])
    print("stress perfect:", data["stress_perfect"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
