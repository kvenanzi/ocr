"""Reimplementation of olmOCR-bench's text, order, baseline and table tests.

Ported from allenai/olmocr `olmocr/bench/tests.py` (v0.4.x) so we can score
without its Chromium/KaTeX dependency. `normalize_text` and the present/absent/
order/baseline logic follow the source line for line. Table matching follows
the documented semantics and is a close, slightly simplified port. Math tests
are not supported (we don't sample those categories).
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

_SPECIAL = re.compile(r"<\|det\|>.*?<\|/det\|>|<\|[^|>]{1,40}\|>|<loc_\d+>", re.S)


def normalize_text(md_content: str) -> str:
    if md_content is None:
        return None
    md_content = re.sub(r"<br/?>", " ", md_content)
    md_content = re.sub(r"\s+", " ", md_content)
    md_content = re.sub(r"\*\*(.*?)\*\*", r"\1", md_content)
    md_content = re.sub(r"__(.*?)__", r"\1", md_content)
    md_content = re.sub(r"</?b>", "", md_content)
    md_content = re.sub(r"</?i>", "", md_content)
    md_content = re.sub(r"\*(.*?)\*", r"\1", md_content)
    md_content = re.sub(r"_(.*?)_", r"\1", md_content)
    md_content = unicodedata.normalize("NFC", md_content)
    replacements = {"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "＿": "_", "–": "-", "—": "-",
                    "‑": "-", "‒": "-", "−": "-", "µ": "μ"}
    for fancy, ascii_char in replacements.items():
        md_content = md_content.replace(fancy, ascii_char)
    return md_content


# ------------------------------------------------------------------ present / absent / order

def _presence(t: dict, md: str) -> bool:
    query = normalize_text(t["text"])
    md = normalize_text(md)
    if not t.get("case_sensitive", True):
        query, md = query.lower(), md.lower()
    first_n, last_n = t.get("first_n"), t.get("last_n")
    if first_n and last_n:
        md = md[:first_n] + md[-last_n:]
    elif first_n:
        md = md[:first_n]
    elif last_n:
        md = md[-last_n:]
    threshold = 1.0 - (t.get("max_diffs", 0) / (len(query) if len(query) > 0 else 1))
    best = fuzz.partial_ratio(query, md) / 100.0
    return best >= threshold if t["type"] == "present" else best < threshold


def _order(t: dict, md: str) -> bool:
    from fuzzysearch import find_near_matches

    md = normalize_text(md)
    before = find_near_matches(normalize_text(t["before"]), md, max_l_dist=t.get("max_diffs", 0))
    after = find_near_matches(normalize_text(t["after"]), md, max_l_dist=t.get("max_diffs", 0))
    return any(b.start < a.start for b in before for a in after)


# ------------------------------------------------------------------ baseline

def _ngram_repeats(text: str, max_n: int = 5) -> list[int]:
    """Consecutive repeats of the trailing n-gram (characters), for n = 1..max_n."""
    text = re.sub(r"\s+", " ", text)
    out = []
    for n in range(1, max_n + 1):
        if len(text) < n:
            out.append(0)
            continue
        tail, count, i = text[-n:], 0, len(text) - n
        while i >= 0 and text[i:i + n] == tail:
            count += 1
            i -= n
        out.append(count)
    return out


_DISALLOWED = re.compile(r"[一-鿿぀-ゟ゠-ヿ\U0001F600-\U0001F64F"
                         r"\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]")


def _baseline(t: dict, content: str) -> bool:
    base_len = len("".join(c for c in content if c.isalnum()).strip())
    if t.get("max_length") is not None:
        return base_len <= t["max_length"]
    if base_len == 0:
        return False
    if any(c > t.get("max_repeats", 30) for c in _ngram_repeats(content)):
        return False
    if t.get("check_disallowed_characters", True) and _DISALLOWED.search(content):
        return False
    return True


# ------------------------------------------------------------------ tables

def _md_tables(md: str) -> list[list[list[str]]]:
    tables, block = [], []
    for line in md.splitlines() + [""]:
        if "|" in line:
            block.append(line)
            continue
        if len(block) >= 2:
            rows = []
            for ln in block:
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) or not c for c in cells):
                    continue
                rows.append(cells)
            if len(rows) >= 2:
                tables.append(rows)
        block = []
    return tables


def _html_tables(md: str) -> list[list[list[str]]]:
    if "<table" not in md.lower():
        return []
    from bs4 import BeautifulSoup

    tables = []
    for table in BeautifulSoup(md, "html.parser").find_all("table"):
        grid: dict[tuple[int, int], str] = {}
        for r, tr in enumerate(table.find_all("tr")):
            c = 0
            for cell in tr.find_all(["td", "th"]):
                while (r, c) in grid:
                    c += 1
                text = cell.get_text(" ", strip=True)
                rs, cs = int(cell.get("rowspan", 1) or 1), int(cell.get("colspan", 1) or 1)
                for dr in range(rs):
                    for dc in range(cs):
                        grid[(r + dr, c + dc)] = text
                c += cs
        if grid:
            n_r = max(k[0] for k in grid) + 1
            n_c = max(k[1] for k in grid) + 1
            tables.append([[grid.get((r, c), "") for c in range(n_c)] for r in range(n_r)])
    return tables


def _table(t: dict, md: str) -> bool:
    tables = _html_tables(md) + ([] if t.get("ignore_markdown_tables") else _md_tables(md))
    if not tables:
        return False
    target = normalize_text(t["cell"])

    def match(ref: str | None, cand: str) -> bool:
        ref = normalize_text(ref)
        thr = max(0.5, 1 - t.get("max_diffs", 0) / max(len(ref), 1))
        return fuzz.ratio(ref, normalize_text(cand)) / 100 >= thr

    for grid in tables:
        n_r = len(grid)
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if not match(target, cell):
                    continue
                ok = True
                for key, (dr, dc) in {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}.items():
                    if t.get(key):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < n_r and 0 <= cc < len(grid[rr]):
                            ok &= match(t[key], grid[rr][cc])
                if ok and t.get("top_heading"):
                    ok = any(match(t["top_heading"], grid[rr][c]) for rr in range(r) if c < len(grid[rr]))
                if ok and t.get("left_heading"):
                    ok = any(match(t["left_heading"], row[cc]) for cc in range(c))
                if ok:
                    return True
    return False


# ------------------------------------------------------------------ entry point

_RUNNERS = {"present": _presence, "absent": _presence, "order": _order, "baseline": _baseline, "table": _table}


def evaluate(tests: list[dict], output: str) -> list[dict]:
    """Run every test for one page. A baseline test is added if the page has none (as the bench does)."""
    output = _SPECIAL.sub("", output or "")
    tests = list(tests)
    if not any(t["type"] == "baseline" for t in tests):
        tests.append({"type": "baseline", "id": "auto_baseline"})
    results = []
    for t in tests:
        runner = _RUNNERS.get(t["type"])
        if runner is None:
            continue
        try:
            passed = bool(runner(t, output))
        except Exception:
            passed = False
        results.append({"id": t.get("id"), "type": t["type"], "passed": passed,
                        "category": "baseline" if t["type"] == "baseline" else t.get("category", "?")})
    return results
