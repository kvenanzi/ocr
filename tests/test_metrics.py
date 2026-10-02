from ocrbench.metrics import normalize
from ocrbench.metrics.loops import has_loop
from ocrbench.metrics.text import score_page


def test_markdown_and_html_are_stripped():
    md = "## Title\n\n**Bold** and *italic* [link](http://x)\n\n| a | b |\n|---|---|\n| 1 | 2 |\n![fig](img.png)"
    assert normalize.diplomatic(md) == "Title Bold and italic link a b 1 2"
    html_table = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    assert normalize.diplomatic(html_table) == "a b 1 2"


def test_grounding_tokens_removed():
    out = "<|ref|>text<|/ref|><|det|>[[10, 20, 30, 40]]<|/det|>\nHello world"
    assert normalize.diplomatic(out) == "text Hello world"


def test_reading_normalisation():
    assert normalize.reading("Stillingﬂeet’s ſhell — “quoted”") == "Stillingfleet's shell - \"quoted\""
    assert normalize.reading("hyphen-\nated word") == "hyphenated word"
    assert normalize.reading("« Je m'endors. »") == normalize.reading("\"Je m'endors.\"")


def test_diplomatic_keeps_historic_characters():
    assert normalize.diplomatic("ſhell") != normalize.diplomatic("shell")


def test_cer_exact_values():
    s = score_page("abcdefghij", "abcdefghiX")
    assert s["cer"] == 0.1
    assert score_page("hello world", "hello world")["cer"] == 0
    # line breaks and formatting are free
    assert score_page("hello\nworld", "# hello   world")["cer"] == 0


def test_wer_and_bag_of_words():
    s = score_page("the quick brown fox", "the quick red fox jumps")
    assert s["wer"] == 0.5          # 1 substitution + 1 insertion over 4 words
    assert s["bow_recall"] == 0.75   # brown missing
    assert s["over_extraction"] == 0.4  # red, jumps out of 5 predicted words


def test_reading_order_hurts_cer_not_bow():
    s = score_page("alpha beta gamma delta", "gamma delta alpha beta")
    assert s["cer"] > 0.3
    assert s["bow_recall"] == 1.0


def test_empty_prediction():
    s = score_page("some text", "")
    assert s["empty"] and s["cer"] == 1.0 and s["bow_recall"] == 0


def test_loop_detection():
    looping = "The species is common. " + "and the and the " * 40
    assert has_loop(looping)
    assert not has_loop(" ".join(f"word{i}" for i in range(500)))
    assert score_page("x", "ok", finish_reason="length")["loop"]
