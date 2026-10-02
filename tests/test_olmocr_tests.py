from ocrbench.metrics.olmocr_tests import evaluate


def _passed(tests, out):
    return {r["id"]: r["passed"] for r in evaluate(tests, out)}


def test_present_absent_order():
    tests = [
        {"id": "p", "type": "present", "text": "The quick brown fox", "max_diffs": 1, "case_sensitive": True},
        {"id": "a", "type": "absent", "text": "Page 12", "max_diffs": 0, "case_sensitive": False},
        {"id": "o", "type": "order", "before": "First section.", "after": "Second section.", "max_diffs": 0},
    ]
    good = "First section. The quick brown fux jumps.\n\nSecond section."
    assert _passed(tests, good) == {"p": True, "a": True, "o": True, "auto_baseline": True}
    bad = "Second section. PAGE 12 First section."
    r = _passed(tests, bad)
    assert not r["p"] and not r["a"] and not r["o"]


def test_baseline_catches_loops_and_empty():
    assert not _passed([], "")["auto_baseline"]
    assert not _passed([], "text " + "." * 40)["auto_baseline"]


def test_table_markdown_and_html():
    t = [{"id": "t", "type": "table", "cell": "0.569", "left_heading": "BO", "max_diffs": 0, "top_heading": "Score"}]
    md = "| Model | Score |\n|---|---|\n| BO | 0.569 |\n| GP | 0.401 |"
    assert _passed(t, md)["t"]
    html = "<table><tr><th>Model</th><th>Score</th></tr><tr><td>BO</td><td>0.569</td></tr></table>"
    assert _passed(t, html)["t"]
    assert not _passed(t, "BO 0.569")["t"]   # plain text has no table structure
