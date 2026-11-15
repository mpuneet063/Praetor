"""Grounded answer generation with Qwen through any OpenAI-compatible endpoint."""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .config import Settings
from .retrieve import RetrievalResult

SYSTEM_PROMPT = """You are Praetor, a legal research assistant for Indian central legislation.

Rules:
1. Answer ONLY from the numbered SOURCES. Never use outside knowledge of the law.
2. End every sentence that states law with its citation in square brackets, e.g. [S2] or [S1][S3].
3. Name the Act and section/article in prose too, e.g. "Under Section 103 of the BNS ... [S1]".
4. If the sources do not contain the answer, reply exactly: "I couldn't find this in the indexed Acts." and nothing else.
5. The Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS) and Bharatiya Sakshya Adhiniyam (BSA) replaced the IPC, CrPC and Indian Evidence Act from 1 July 2024. If both old and new provisions appear, give the current (new) law first and mention the old section.
6. Be concise: a direct answer first, then key conditions or exceptions as short bullet points.
7. Do not give personal legal advice."""

REFUSAL = "I couldn't find this in the indexed Acts."
DISCLAIMER = "Praetor is a research tool, not legal advice. Check the cited provisions at the source."


def format_sources(result: RetrievalResult) -> str:
    blocks = []
    for i, h in enumerate(result.hits, 1):
        c = h.chunk
        status = f" [status: {c.status}]" if c.status and c.status.lower() != "in_force" else ""
        blocks.append(f"[S{i}] {c.act_title} — {c.label}: {c.section_title}{status}\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, result: RetrievalResult) -> list[dict]:
    notes = ""
    if result.mappings:
        notes = "Old→new code correspondence detected: " + "; ".join(m.describe() for m in result.mappings) + "\n\n"
    # "/no_think" is Qwen3's soft switch to skip the reasoning trace (faster, fewer tokens).
    user = f"{notes}SOURCES:\n{format_sources(result)}\n\nQUESTION: {question} /no_think"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_CITE_RE = re.compile(r"\[S(\d+)\]")


def strip_thinking(text: str) -> str:
    """Qwen3 may emit <think>...</think>; drop it (including an unterminated block while streaming)."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.lstrip()


@dataclass
class CitationReport:
    cited: list[int]          # source numbers actually cited
    invalid: list[int]        # cited numbers with no matching source
    uncited_sentences: int    # sentences with no citation (excluding short bullets/headers)
    total_sentences: int

    @property
    def coverage(self) -> float:
        return 1.0 - self.uncited_sentences / self.total_sentences if self.total_sentences else 1.0


def check_citations(answer: str, n_sources: int) -> CitationReport:
    nums = [int(n) for n in _CITE_RE.findall(answer)]
    invalid = sorted({n for n in nums if n < 1 or n > n_sources})
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(s.split()) >= 6]
    uncited = sum(1 for s in sentences if not _CITE_RE.search(s))
    return CitationReport(sorted(set(nums) - set(invalid)), invalid, uncited, len(sentences))


def should_refuse(result: RetrievalResult, settings: Settings, reranked: bool) -> bool:
    if not result.hits:
        return True
    return reranked and result.best_score < settings.min_rerank_score


class LLM:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.s = settings
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "none")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield the visible answer text as it grows (thinking stripped)."""
        resp = self.client.chat.completions.create(
            model=self.s.llm_model, messages=messages, temperature=self.s.temperature,
            max_tokens=self.s.max_tokens, stream=True,
        )
        raw = ""
        for event in resp:
            if not event.choices:
                continue
            raw += event.choices[0].delta.content or ""
            yield strip_thinking(raw)
