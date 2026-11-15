"""Build, save and load the search index (chunks + dense vectors + BM25)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import bm25s
import numpy as np

from .config import BGE_QUERY_PREFIX, EMBED_MODEL, RERANK_MODEL
from .data import Chunk

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """BGE-small via fastembed (ONNX on CPU, no torch needed)."""

    def __init__(self, model: str = EMBED_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts, batch_size=64)), dtype=np.float32)
        return _normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(next(iter(self.model.embed([BGE_QUERY_PREFIX + text]))), dtype=np.float32)
        return _normalize(vec[None])[0]


class FastReranker:
    """MiniLM cross-encoder via fastembed. Returns raw logits (higher = more relevant)."""

    def __init__(self, model: str = RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        return np.array(list(self.model.rerank(query, texts)), dtype=np.float32)


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def passage_text(c: Chunk) -> str:
    """Text that gets indexed: context header + body, so 'BNS 103 murder' matches lexically."""
    head = f"{c.act_title} | {c.label} | {c.section_title}"
    if c.chapter:
        head += f" | {c.chapter}"
    return f"{head}\n{c.text}"


def tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", show_progress=False)


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray  # (n, d) float32, L2-normalised
    bm25: bm25s.BM25
    meta: dict

    # ---- persistence -------------------------------------------------------
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(out_dir / "vectors.npy", self.vectors.astype(np.float16))
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, in_dir: Path) -> "Index":
        chunks = []
        with open(in_dir / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk(**json.loads(line)))
        vectors = np.load(in_dir / "vectors.npy").astype(np.float32)
        meta = json.loads((in_dir / "meta.json").read_text())
        # BM25 rebuilds in about a second for this corpus size, so it isn't persisted.
        bm25 = bm25s.BM25()
        bm25.index(tokenize([passage_text(c) for c in chunks]), show_progress=False)
        return cls(chunks, vectors, bm25, meta)

    @staticmethod
    def exists(in_dir: Path) -> bool:
        return all((in_dir / n).exists() for n in ("chunks.jsonl", "vectors.npy", "meta.json"))


def build_index(chunks: list[Chunk], embedder: Embedder, extra_meta: dict | None = None) -> Index:
    if not chunks:
        raise ValueError("No chunks to index — did the Act filter match anything?")
    texts = [passage_text(c) for c in chunks]
    t0 = time.time()
    vectors = embedder.embed_docs(texts)
    log.info("Embedded %d chunks in %.1fs", len(texts), time.time() - t0)
    bm25 = bm25s.BM25()
    bm25.index(tokenize(texts), show_progress=False)
    meta = {"n_chunks": len(chunks), "dim": int(vectors.shape[1]), "embed_model": EMBED_MODEL,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta.update(extra_meta or {})
    return Index(chunks, vectors, bm25, meta)
