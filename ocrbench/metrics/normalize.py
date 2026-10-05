"""Turn engine output into comparable plain text.

Engines disagree on output format: Tesseract emits plain lines, VLMs emit
Markdown, HTML tables, LaTeX, or grounding tokens. Before any scoring, all
output goes through `to_plain`, so formatting choices cost nothing and only the
recognised characters count.

Two comparison modes, following FineBooks:
  diplomatic  every character as printed (long-s, ligatures, curly quotes count)
  reading     NFKC + long-s -> s + quotes/dashes unified + line-end hyphens joined.
              This asks "did it read the words right?", which is what most users
              of OCR output (search, LLM training, RAG) care about.
"""
from __future__ import annotations

import html
import re
import unicodedata

_SPECIAL_BLOCKS = [
    re.compile(r"<\|det\|>.*?<\|/det\|>", re.S),      # DeepSeek-OCR boxes
    re.compile(r"<\|box_start\|>.*?<\|box_end\|>", re.S),
    re.compile(r"<loc_\d+>"),                          # DocTags locations
    re.compile(r"<\|[^|>]{1,40}\|>"),                  # any remaining <|token|>
]
_HTML_BREAKS = re.compile(r"<\s*(br|/p|/tr|/li|/h[1-6]|/div|/table|/caption)\s*/?>", re.I)
_HTML_CELLS = re.compile(r"<\s*/?\s*(td|th)[^>]*>", re.I)
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_-]*(\s[^<>]*)?/?>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?", re.M)
_MD_BULLET = re.compile(r"^\s*[-*+•]\s+", re.M)
_MD_RULE = re.compile(r"^\s*([-*_=]\s*){3,}$", re.M)
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", re.M)
_MD_FENCE = re.compile(r"^\s*```[^\n]*$", re.M)
_MD_EMPH = re.compile(r"(\*\*|__|\*|~~)(?=\S)(.+?)(?<=\S)\1", re.S)
_LATEX_DELIMS = re.compile(r"\\\(|\\\)|\\\[|\\\]|\$\$?")
_LATEX_WRAP = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|emph|operatorname)\s*\{([^{}]*)\}")
_WS = re.compile(r"\s+")


def to_plain(text: str) -> str:
    """Strip Markdown / HTML / LaTeX / grounding markup, keeping text in reading order."""
    if not text:
        return ""
    for pat in _SPECIAL_BLOCKS:
        text = pat.sub(" ", text)
    text = _HTML_BREAKS.sub("\n", text)
    text = _HTML_CELLS.sub(" ", text)
    text = _HTML_TAG.sub("", text)
    text = html.unescape(text)
    text = _MD_FENCE.sub("", text)
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_TABLE_SEP.sub("", text)
    text = _MD_RULE.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_QUOTE.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_EMPH.sub(r"\2", text)
    text = text.replace("|", " ")
    text = _LATEX_WRAP.sub(r"\1", text)
    text = _LATEX_DELIMS.sub("", text)
    return text


_CHAR_MAP = str.maketrans({
    "ſ": "s", "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "\u00ad": None, "\u200b": None, "\ufeff": None,
})
_HYPHEN_BREAK = re.compile(r"(\w)[-¬]\s*\n\s*(\w)")


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def diplomatic(text: str, markup: bool = True) -> str:
    return collapse_ws(to_plain(text) if markup else text)


def reading(text: str, markup: bool = True) -> str:
    text = to_plain(text) if markup else text
    text = unicodedata.normalize("NFKC", text)
    # French typography puts spaces inside « » and before ; : ! ? — treat them as optional.
    text = re.sub(r"[\s\u202f]+([;:!?»])", r"\1", text)
    text = re.sub(r"«[\s\u202f]+", "«", text)
    text = text.translate(_CHAR_MAP)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return collapse_ws(text)


_WORD = re.compile(r"\w+", re.U)


def words(text: str) -> list[str]:
    """Case-folded word tokens with punctuation removed (for bag-of-words metrics)."""
    return [w.casefold() for w in _WORD.findall(text)]
