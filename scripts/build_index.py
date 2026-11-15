"""Build the Praetor index.

    python scripts/build_index.py                 # download from the Vaquill mirror / HF
    python scripts/build_index.py --parquet f.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praetor.build import build_from_parquet, download_central  # noqa: E402
from praetor.config import INDEX_DIR, ROOT, Settings  # noqa: E402
from praetor.index import FastEmbedder  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, help="local in_central_legislation.parquet")
    ap.add_argument("--out", type=Path, default=INDEX_DIR)
    args = ap.parse_args()

    parquet = args.parquet or download_central(ROOT / ".cache" / "in_central_legislation.parquet")
    index = build_from_parquet(parquet, FastEmbedder(), Settings())
    index.save(args.out)
    print(f"Saved {index.meta['n_chunks']:,} chunks to {args.out}")
    for act, n in index.meta["coverage"].items():
        print(f"  {act:<12} {n:>6}")


if __name__ == "__main__":
    main()
