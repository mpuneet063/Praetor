"""Retrieval ablation: Recall@k and MRR for BM25 / dense / hybrid / hybrid+rerank.

    python scripts/eval_retrieval.py            # prints a markdown table, writes eval/results.md

A question counts as a hit if ANY gold (act, section) appears in the top-k.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praetor.config import INDEX_DIR, ROOT, Settings  # noqa: E402
from praetor.index import FastEmbedder, FastReranker, Index  # noqa: E402
from praetor.retrieve import Retriever  # noqa: E402

MODES = ["bm25", "dense", "hybrid", "hybrid+rerank"]
KS = [1, 3, 5, 10]


def first_hit_rank(hits, gold) -> int | None:
    gold = {(a, s.upper()) for a, s in gold}
    for r, h in enumerate(hits, 1):
        if (h.chunk.act_key, h.chunk.section.upper()) in gold:
            return r
    return None


def evaluate(retriever: Retriever, questions: list[dict]) -> dict[str, dict]:
    out = {}
    for mode in MODES:
        ranks, t0 = [], time.time()
        for q in questions:
            res = retriever.search(q["q"], mode=mode, top_k=max(KS))
            ranks.append(first_hit_rank(res.hits, q["gold"]))
        n = len(questions)
        row = {f"R@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
        row["MRR@10"] = sum(1 / r for r in ranks if r) / n
        row["ms/q"] = 1000 * (time.time() - t0) / n
        out[mode] = row
    return out


def to_markdown(results: dict[str, dict], n: int) -> str:
    cols = [f"R@{k}" for k in KS] + ["MRR@10", "ms/q"]
    lines = [f"Retrieval evaluation on {n} questions (`eval/questions.jsonl`)\n",
             "| Mode | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mode, row in results.items():
        cells = [f"{row[c]:.0f}" if c == "ms/q" else f"{row[c]:.3f}" for c in cols]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX_DIR)
    ap.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    args = ap.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    retriever = Retriever(Index.load(args.index), FastEmbedder(), FastReranker(), Settings())
    md = to_markdown(evaluate(retriever, questions), len(questions))
    print(md)
    (ROOT / "eval" / "results.md").write_text(md)


if __name__ == "__main__":
    main()
