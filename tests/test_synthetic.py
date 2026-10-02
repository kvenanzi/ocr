from ocrbench.data import synthetic
from ocrbench.data.base import read_manifest
from ocrbench.data.iam import detokenize


def test_render_is_deterministic():
    a = synthetic.render_base_page(1, seed=7)
    b = synthetic.render_base_page(1, seed=7)
    assert a[1] == b[1] and a[0].tobytes() == b[0].tobytes()


def test_prepare_writes_every_level(tmp_path):
    samples = synthetic.prepare(tmp_path, n_base=2, seed=3, levels=("clean", "dpi75", "phone_photo"))
    assert len(samples) == 6
    assert {s.meta["level"] for s in samples} == {"clean", "dpi75", "phone_photo"}
    assert all((tmp_path / s.image).exists() for s in samples)
    assert read_manifest(tmp_path / "manifest.jsonl")[0].gt_text == samples[0].gt_text


def test_all_levels_run():
    img, *_ = synthetic.render_base_page(0, seed=1)
    small = img.resize((400, 520))
    for level in synthetic.LEVELS:
        assert synthetic.degrade(small, level, seed=0).size[0] > 0


def test_iam_detokenize():
    assert detokenize("it 's a ( good ) start , \" ok \" .") == "it's a (good) start, \"ok\"."
