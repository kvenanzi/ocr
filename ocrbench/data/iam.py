"""Handwriting track: IAM handwritten text lines (Teklia/IAM-line, test split).

Each sample is a single handwritten line. Classic engines are expected to do
badly here; that contrast is the point of the track.
"""
from __future__ import annotations

import io
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from .base import Sample, write_manifest

REPO = "Teklia/IAM-line"


def detokenize(text: str) -> str:
    """IAM transcriptions are tokenised ("it 's a good start ."); restore normal spacing."""
    text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+('(?:s|t|re|ll|ve|d|m)\b|n't\b)", r"\1", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def prepare(out_dir: Path, n: int, seed: int) -> list[Sample]:
    table = pq.read_table(hf_hub_download(REPO, "data/test.parquet", repo_type="dataset"))
    picked = sorted(random.Random(seed).sample(range(table.num_rows), min(n, table.num_rows)))
    rows = table.take(picked).to_pylist()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    samples = []
    for i, row in zip(picked, rows):
        rel = f"images/iam_{i:05d}.png"
        Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(out_dir / rel)
        samples.append(Sample(id=f"iam_{i:05d}", track="handwriting", image=rel,
                              gt_text=detokenize(row["text"]), meta={"raw_gt": row["text"]}))
    write_manifest(out_dir, samples)
    return samples
