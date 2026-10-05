"""Synthetic stress track: render known text, then degrade it step by step.

Every base page is rendered once, clean, at 300 DPI. It is then pushed through
each level of LEVELS, so the report can draw a robustness curve showing how
fast each engine's CER rises as the image gets worse.
"""
from __future__ import annotations

import glob
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import corpus
from .base import Sample, write_manifest

DPI = 300
PAGE_W, PAGE_H = int(8.5 * DPI), int(11 * DPI)
MARGIN = int(0.75 * DPI)
LAYOUTS = ("prose", "two_column", "table", "invoice")

# Ordered roughly from easiest to hardest; the report keeps this order on the x-axis.
LEVELS = (
    "clean", "dpi150", "dpi75", "blur", "noise", "rotate3", "rotate8",
    "jpeg_q10", "low_contrast", "photocopy", "phone_photo",
)

_FONT_GLOBS = [
    "/usr/share/fonts/truetype/liberation*/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation*/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "C:/Windows/Fonts/times.ttf",
]


def available_fonts() -> list[str]:
    found = []
    for pattern in _FONT_GLOBS:
        found += sorted(glob.glob(pattern))
    return found


def _font(path: str | None, pt: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    px = round(pt / 72 * DPI)
    if path is None:
        return ImageFont.load_default(size=px)
    if bold:
        for cand in (path.replace("-Regular", "-Bold"), path.replace(".ttf", "-Bold.ttf")):
            if cand != path and Path(cand).exists():
                path = cand
                break
    return ImageFont.truetype(path, px)


def _wrap(text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class _Page:
    def __init__(self):
        self.img = Image.new("L", (PAGE_W, PAGE_H), 255)
        self.draw = ImageDraw.Draw(self.img)

    def text_block(self, text: str, font, x: int, y: int, width: int, spacing: float = 1.35) -> tuple[int, list[str]]:
        lines = _wrap(text, font, width)
        step = int(font.size * spacing)
        for line in lines:
            self.draw.text((x, y), line, font=font, fill=0)
            y += step
        return y, lines


def _render_prose(rng: random.Random, font_path) -> tuple[Image.Image, str]:
    page, body, head = _Page(), _font(font_path, 12), _font(font_path, 18, bold=True)
    width = PAGE_W - 2 * MARGIN
    heading = rng.choice(corpus.HEADINGS)
    y, _ = page.text_block(heading, head, MARGIN, MARGIN, width)
    y += body.size
    gt = [heading]
    for para in rng.sample(corpus.PARAGRAPHS, 7):
        y, lines = page.text_block(para, body, MARGIN, y, width)
        gt.append("\n".join(lines))
        y += body.size // 2
    return page.img, "\n\n".join(gt)


def _render_two_column(rng: random.Random, font_path) -> tuple[Image.Image, str]:
    page, body, head = _Page(), _font(font_path, 10), _font(font_path, 16, bold=True)
    gutter = int(0.35 * DPI)
    col_w = (PAGE_W - 2 * MARGIN - gutter) // 2
    heading = rng.choice(corpus.HEADINGS)
    y0, _ = page.text_block(heading, head, MARGIN, MARGIN, PAGE_W - 2 * MARGIN)
    y0 += body.size
    paras = rng.sample(corpus.PARAGRAPHS, 6)
    gt = [heading]
    # Reading order is the whole left column, then the whole right column.
    for col, chunk in enumerate((paras[:3], paras[3:])):
        x, y = MARGIN + col * (col_w + gutter), y0
        for para in chunk:
            y, lines = page.text_block(para, body, x, y, col_w)
            gt.append("\n".join(lines))
            y += body.size // 2
    return page.img, "\n\n".join(gt)


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _table(page: _Page, font, x: int, y: int, rows: list[list[str]], col_w: list[int]) -> int:
    row_h = int(font.size * 1.9)
    total_w = sum(col_w)
    for r, row in enumerate(rows):
        cx = x
        for c, cell in enumerate(row):
            page.draw.text((cx + 15, y + (row_h - font.size) // 2), cell, font=font, fill=0)
            cx += col_w[c]
        page.draw.line([(x, y), (x + total_w, y)], fill=0, width=3 if r <= 1 else 1)
        y += row_h
    page.draw.line([(x, y), (x + total_w, y)], fill=0, width=3)
    cx = x
    for w in [0, *col_w]:
        cx += w
        page.draw.line([(cx, y - row_h * len(rows)), (cx, y)], fill=0, width=1)
    return y


def _line_items(rng: random.Random, n: int) -> tuple[list[list[str]], float]:
    rows, subtotal = [], 0.0
    for item in rng.sample(corpus.ITEMS, n):
        qty, price = rng.randint(1, 40), rng.randint(150, 48000) / 100
        subtotal += qty * price
        rows.append([item, str(qty), _money(price), _money(qty * price)])
    return rows, subtotal


def _render_table(rng: random.Random, font_path) -> tuple[Image.Image, str]:
    page, body, head = _Page(), _font(font_path, 10.5), _font(font_path, 16, bold=True)
    title = f"Table {rng.randint(2, 19)}. Conservation supplies, quarter {rng.randint(1, 4)}"
    y, _ = page.text_block(title, head, MARGIN, MARGIN, PAGE_W - 2 * MARGIN)
    y += body.size
    header = ["Item", "Qty", "Unit price", "Amount"]
    items, subtotal = _line_items(rng, 8)
    rows = [header, *items, ["Total", "", "", _money(subtotal)]]
    _table(page, body, MARGIN, y, rows, [1000, 220, 400, 430])
    gt = [title] + [" ".join(c for c in row if c) for row in rows]
    return page.img, "\n".join(gt)


def _render_invoice(rng: random.Random, font_path) -> tuple[Image.Image, str]:
    page, body, head = _Page(), _font(font_path, 10.5), _font(font_path, 20, bold=True)
    width = PAGE_W - 2 * MARGIN
    company = rng.choice(corpus.COMPANIES)
    inv_no = f"INV-{rng.randint(2019, 2026)}-{rng.randint(0, 99999):05d}"
    date = f"{rng.randint(1, 28)} {rng.choice(['January', 'March', 'June', 'August', 'November'])} {rng.randint(2019, 2026)}"
    lines = [
        f"Invoice No: {inv_no}",
        f"Date: {date}",
        f"Bill to: {rng.choice(corpus.PEOPLE)}",
        f"Account: {rng.randint(100, 999)}-{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJK')}{rng.randint(10, 99)}",
        f"Reference: PO {rng.randint(10000, 99999)}/{rng.choice(['A', 'B', 'C'])}",
    ]
    y, _ = page.text_block(company, head, MARGIN, MARGIN, width)
    y += body.size
    for line in lines:
        y, _ = page.text_block(line, body, MARGIN, y, width)
    y += body.size
    items, subtotal = _line_items(rng, 5)
    rows = [["Description", "Qty", "Unit price", "Amount"], *items]
    y = _table(page, body, MARGIN, y, rows, [1000, 220, 400, 430]) + body.size
    tax_rate = rng.choice([5.0, 7.5, 8.25, 19.0, 20.0])
    tax = subtotal * tax_rate / 100
    totals = [f"Subtotal: {_money(subtotal)}", f"Tax ({tax_rate:g}%): {_money(tax)}",
              f"Total due: {_money(subtotal + tax)}"]
    for line in totals:
        y, _ = page.text_block(line, body, MARGIN + 1100, y, width - 1100)
    gt = [company, *lines, *[" ".join(r) for r in rows], *totals]
    return page.img, "\n".join(gt)


_RENDERERS = {"prose": _render_prose, "two_column": _render_two_column,
              "table": _render_table, "invoice": _render_invoice}


# ---------------------------------------------------------------- degradations

def _scale(img: Image.Image, f: float) -> Image.Image:
    return img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)


def _noise(img: Image.Image, rng: np.random.Generator, sigma: float, sp: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) + rng.normal(0, sigma, (img.height, img.width))
    mask = rng.random(a.shape)
    a[mask < sp / 2] = 0
    a[mask > 1 - sp / 2] = 255
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _perspective_coeffs(src, dst):
    """Coefficients for Image.transform(PERSPECTIVE) mapping dst quad -> src quad."""
    m = []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    return np.linalg.solve(np.array(m, dtype=float), np.array(src, dtype=float).reshape(8))


def degrade(img: Image.Image, level: str, seed: int) -> Image.Image:
    """Apply one degradation level. JPEG artifacts are baked into the returned pixels."""
    rng = np.random.default_rng(seed)
    if level == "clean":
        return img
    if level == "dpi150":
        return _scale(img, 0.5)
    if level == "dpi75":
        return _scale(img, 0.25)
    if level == "blur":
        return img.filter(ImageFilter.GaussianBlur(3.5))
    if level == "noise":
        return _noise(img, rng, sigma=55, sp=0.02)
    if level == "rotate3":
        return img.rotate(3, resample=Image.BICUBIC, expand=True, fillcolor=255)
    if level == "rotate8":
        return img.rotate(-8, resample=Image.BICUBIC, expand=True, fillcolor=255)
    if level == "jpeg_q10":
        buf = io.BytesIO()
        _scale(img, 0.5).save(buf, "JPEG", quality=10)
        return Image.open(io.BytesIO(buf.getvalue()))
    if level == "low_contrast":
        return img.point(lambda p: 175 + p * 40 // 255)
    if level == "photocopy":
        a = np.asarray(img.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
        a += rng.normal(0, 35, a.shape)
        a = np.where(a < 150, 0, 255).astype(np.float32)
        specks = rng.random(a.shape) < 0.004
        a[specks] = 0
        # dark binding shadow fading in from the left edge
        shadow = np.clip(1 - np.arange(a.shape[1]) / (0.08 * a.shape[1]), 0, 1) * 200
        a = np.clip(a - shadow[None, :], 0, 255)
        return Image.fromarray(a.astype(np.uint8))
    if level == "phone_photo":
        small = _scale(img, 0.5)
        w, h = small.size
        j = lambda s: rng.uniform(0.02, 0.06) * s  # noqa: E731
        dst = [(j(w), j(h)), (w - j(w), j(h) * 0.3), (w - j(w) * 0.5, h - j(h)), (j(w) * 0.4, h - j(h) * 0.6)]
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        warped = small.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(src, dst),
                                 Image.BICUBIC, fillcolor=90)
        yy, xx = np.mgrid[0:h, 0:w]
        light = 0.65 + 0.35 * np.exp(-(((xx - 0.3 * w) / (0.9 * w)) ** 2 + ((yy - 0.25 * h) / (0.9 * h)) ** 2))
        a = np.asarray(warped, dtype=np.float32) * light
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    raise ValueError(f"unknown degradation level {level!r}")


def render_base_page(index: int, seed: int) -> tuple[Image.Image, str, str, str]:
    rng = random.Random(seed * 1000 + index)
    fonts = available_fonts() or [None]
    layout = LAYOUTS[index % len(LAYOUTS)]
    font_path = fonts[index % len(fonts)]
    img, gt = _RENDERERS[layout](rng, font_path)
    return img, gt, layout, Path(font_path).stem if font_path else "default"


def prepare(out_dir: Path, n_base: int, seed: int, levels=LEVELS) -> list[Sample]:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n_base):
        img, gt, layout, font = render_base_page(i, seed)
        for li, level in enumerate(levels):
            rel = f"images/syn{i:03d}_{layout}__{level}.png"
            degrade(img, level, seed=seed * 7919 + i * 101 + li).convert("L").save(out_dir / rel)
            samples.append(Sample(
                id=f"syn{i:03d}_{layout}__{level}", track="stress", image=rel, gt_text=gt,
                meta={"base": f"syn{i:03d}", "layout": layout, "level": level, "font": font},
            ))
    write_manifest(out_dir, samples)
    return samples
