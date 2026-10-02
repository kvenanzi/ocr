"""Historical books track: FineBooks BHL IMPACT ground truth (CC-BY 3.0).

https://huggingface.co/datasets/finebooks/bhl-impact-gt
Expert transcriptions (~99.95% accurate) of six natural-history books, 1708-1913.
We download only metadata.parquet plus the sampled page images, not the full ~390 MB.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from .base import Sample, stratified_sample, write_manifest

REPO = "finebooks/bhl-impact-gt"
LANGUAGE = {
    "birdsofgreatbrit02butl": "en", "conchologiaiconi05reev": "en", "daschitinskelett00prel": "de",
    "histoirenaturell10cuvi": "fr", "pisciumquerelaee00sche": "la", "trudyrusskagoent161881russ": "fr+de",
}
MIN_CHARS = 200   # skip plate pages and near-empty pages


def prepare(out_dir: Path, n: int, seed: int) -> list[Sample]:
    meta = pd.read_parquet(hf_hub_download(REPO, "metadata.parquet", repo_type="dataset"),
                           columns=["file_name", "PageID", "BarCode", "text"])
    meta = meta[meta.text.str.len() >= MIN_CHARS]
    groups = {bc: g.index.tolist() for bc, g in meta.groupby("BarCode")}
    picked = stratified_sample(groups, n, random.Random(seed))

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    samples = []
    for idx in sorted(picked):
        row = meta.loc[idx]
        src = hf_hub_download(REPO, row.file_name, repo_type="dataset")
        rel = f"images/{Path(row.file_name).stem}.png"
        Image.open(src).convert("RGB").save(out_dir / rel)
        samples.append(Sample(
            id=Path(row.file_name).stem, track="books", image=rel, gt_text=row.text,
            meta={"book": row.BarCode, "lang": LANGUAGE.get(row.BarCode, "?"), "page_id": int(row.PageID)},
        ))
    write_manifest(out_dir, samples)
    return samples
