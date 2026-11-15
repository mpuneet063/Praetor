"""Download the corpus and build the index. Used by scripts/build_index.py and by the app
as a fallback when no prebuilt index is present."""
from __future__ import annotations

import logging
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .config import CENTRAL_PARQUET_URL, DATASET_VERSION, HF_CENTRAL_FILE, HF_DATASET, Settings
from .data import coverage_report, load_chunks
from .index import Embedder, Index, build_index

log = logging.getLogger(__name__)


def download_central(dest: Path) -> Path:
    """Fetch in_central_legislation.parquet from the Vaquill mirror, falling back to Hugging Face."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        log.info("Downloading %s", CENTRAL_PARQUET_URL)
        req = urllib.request.Request(CENTRAL_PARQUET_URL, headers={"User-Agent": "praetor/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest
    except Exception as e:  # noqa: BLE001
        log.warning("Mirror download failed (%s); trying Hugging Face", e)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_DATASET, HF_CENTRAL_FILE, repo_type="dataset")
    shutil.copy(path, dest)
    return dest


def build_from_parquet(parquet: Path, embedder: Embedder, settings: Settings,
                       progress: Callable[[str], None] = log.info) -> Index:
    progress("Reading and filtering the central legislation file…")
    chunks = load_chunks(parquet, settings.chunk_words, settings.chunk_overlap)
    cov = coverage_report(chunks)
    missing = [k for k, v in cov.items() if v == 0]
    progress(f"{len(chunks):,} chunks from {sum(1 for v in cov.values() if v)} Acts"
             + (f" (not found: {', '.join(missing)})" if missing else ""))
    progress("Embedding passages (a few minutes on CPU)…")
    return build_index(chunks, embedder, {"dataset": f"{HF_DATASET}@{DATASET_VERSION}", "coverage": cov})
