"""Hybrid retrieval: BM25 + dense, fused with RRF, then cross-encoder rerank."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .data import Chunk
from .index import Embedder, Index, Reranker, passage_text, tokenize
from .mapping import Mapping, SectionRef, expand_query


@dataclass
class Hit:
    chunk: Chunk
    score: float                 # final score (rerank logit, or fused score if no reranker)
    ranks: dict = field(default_factory=dict)  # per-retriever ranks for the debug view


@dataclass
class RetrievalResult:
    query: str
    rewritten: str
    hits: list[Hit]
    mappings: list[Mapping]
    refs: list[SectionRef]

    @property
    def best_score(self) -> float:
        return max((h.score for h in self.hits), default=float("-inf"))


def rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
    return scores


class Retriever:
    def __init__(self, index: Index, embedder: Embedder, reranker: Reranker | None, settings: Settings):
        self.index, self.embedder, self.reranker, self.s = index, embedder, reranker, settings
        self._by_ref: dict[tuple[str, str], list[int]] = {}
        for i, c in enumerate(index.chunks):
            self._by_ref.setdefault((c.act_key, c.section.upper()), []).append(i)

    # ---- individual retrievers ---------------------------------------------
    def bm25(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[int]:
        n = len(self.index.chunks)
        docs, scores = self.index.bm25.retrieve(tokenize([query]), k=min(n, k * 3 if allowed is not None else k),
                                                show_progress=False)
        out = [int(d) for d, s in zip(docs[0], scores[0]) if s > 0]
        if allowed is not None:
            out = [d for d in out if allowed[d]]
        return out[:k]

    def dense(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[int]:
        sims = self.index.vectors @ self.embedder.embed_query(query)
        if allowed is not None:
            sims = np.where(allowed, sims, -np.inf)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        return [int(i) for i in top[np.argsort(-sims[top])] if np.isfinite(sims[i])]

    def exact(self, refs: list[SectionRef], allowed: np.ndarray | None = None) -> list[int]:
        out: list[int] = []
        for r in refs:
            out += self._by_ref.get((r.act, r.section.upper()), [])
        return [i for i in out if allowed is None or allowed[i]]

    # ---- full pipeline -----------------------------------------------------
    def search(self, query: str, acts: list[str] | None = None, mode: str = "hybrid+rerank",
               top_k: int | None = None) -> RetrievalResult:
        """mode: bm25 | dense | hybrid | hybrid+rerank"""
        s, top_k = self.s, top_k or self.s.top_k
        rewritten, refs, mappings = expand_query(query)
        allowed = None
        if acts:
            allowed = np.array([c.act_key in acts for c in self.index.chunks])

        lists: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            lists["bm25"] = self.bm25(rewritten, s.bm25_k, allowed)
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            lists["dense"] = self.dense(rewritten, s.dense_k, allowed)
        exact = self.exact(refs, allowed)
        if exact:  # explicitly cited sections always enter the candidate pool, ranked first
            lists["exact"] = exact

        fused = rrf(list(lists.values()), s.rrf_k)
        cand = sorted(fused, key=fused.get, reverse=True)[: max(s.fuse_k, len(exact))]
        rank_of = {name: {idx: r for r, idx in enumerate(lst)} for name, lst in lists.items()}

        if mode == "hybrid+rerank" and self.reranker is not None and cand:
            texts = [passage_text(self.index.chunks[i]) for i in cand]
            scores = self.reranker.score(query, texts)
            # Explicitly referenced sections get a boost so "what does IPC 420 say" returns it.
            exact_set = set(exact)
            scores = np.array([sc + (3.0 if i in exact_set else 0.0) for sc, i in zip(scores, cand)])
            order = np.argsort(-scores)
            ranked = [(cand[j], float(scores[j])) for j in order]
        else:
            ranked = [(i, fused[i]) for i in cand]

        hits = [
            Hit(self.index.chunks[i], sc, {n: rank_of[n].get(i) for n in rank_of})
            for i, sc in ranked[:top_k]
        ]
        return RetrievalResult(query, rewritten, hits, mappings, refs)
