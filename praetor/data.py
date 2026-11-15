"""Load the Open India Law central-legislation parquet, keep the configured Acts, chunk them."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq

from .config import ACTS, ActSpec

log = logging.getLogger(__name__)

# The dataset card documents these names; alternates guard against schema drift.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "act_id": ["act_id", "id", "doc_id"],
    "title": ["title", "act_title", "act_name", "short_title"],
    "year": ["year", "act_year", "enactment_year"],
    "chapter": ["chapter", "chapter_title", "part"],
    "section_number": ["section_number", "section_no", "section", "provision_number"],
    "section_title": ["section_title", "heading", "section_heading", "provision_title"],
    "text": ["text", "content", "section_text", "chunk_text", "body"],
    "act_status": ["act_status", "status"],
    "section_status": ["section_status"],
    "source_url": ["source_url", "url", "mirror_url"],
}


@dataclass
class Chunk:
    chunk_id: str
    act_key: str
    act_title: str
    year: int | None
    chapter: str
    section: str
    section_title: str
    text: str
    status: str
    source_url: str

    @property
    def label(self) -> str:
        unit = "Article" if self.act_key == "Constitution" else "Section"
        return f"{self.act_key} {unit} {self.section}".strip()

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_columns(names: list[str]) -> dict[str, str | None]:
    lower = {n.lower(): n for n in names}
    out: dict[str, str | None] = {}
    for field, cands in COLUMN_CANDIDATES.items():
        out[field] = next((lower[c] for c in cands if c in lower), None)
    missing = [f for f in ("title", "text") if out[f] is None]
    if missing:
        raise ValueError(f"Parquet is missing required columns {missing}; found {names}")
    return out


_YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


def _candidate_years(title: str, year) -> set[int]:
    ys = {int(y) for y in _YEAR_RE.findall(title or "")}
    if year not in (None, "") and str(year).strip()[:4].isdigit():
        ys.add(int(str(year).strip()[:4]))
    return ys


def match_act(title: str, year) -> ActSpec | None:
    t = (title or "").lower()
    for spec in ACTS:
        if spec.title.lower() not in t:
            continue
        if spec.year is not None:
            ys = _candidate_years(title, year)
            # Short-title year and enactment year can differ by one (CrPC 1973 is Act 2 of 1974).
            if ys and not any(abs(y - spec.year) <= 1 for y in ys):
                continue
        # Avoid matching amendment Acts, e.g. "Indian Penal Code (Amendment) Act".
        if "amendment" in t and "amendment" not in spec.title.lower():
            continue
        return spec
    return None


_SEC_RE = re.compile(r"(\d+[A-Z]{0,3})", re.I)


def normalize_section(raw) -> str:
    """'Section 304B.' -> '304B'; 'Article 21' -> '21'; leaves non-numeric labels intact."""
    s = str(raw or "").strip()
    m = _SEC_RE.search(s)
    return m.group(1).upper() if m else s


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text.strip()]
    out, step = [], max(1, size - overlap)
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def load_chunks(parquet_path: str | Path, chunk_words: int = 320, overlap: int = 60) -> list[Chunk]:
    """Stream the parquet in batches, keeping only rows that belong to configured Acts."""
    pf = pq.ParquetFile(parquet_path)
    cols = resolve_columns(pf.schema_arrow.names)
    log.info("Parquet columns: %s", pf.schema_arrow.names)
    log.info("Column mapping: %s", cols)
    wanted = [c for c in cols.values() if c]

    chunks: list[Chunk] = []
    seen: set[tuple] = set()
    for batch in pf.iter_batches(columns=wanted, batch_size=20_000):
        d = batch.to_pydict()
        n = len(d[cols["title"]])
        get = lambda f, i: (d[cols[f]][i] if cols[f] else None)  # noqa: E731
        for i in range(n):
            title, year = get("title", i), get("year", i)
            spec = match_act(title, year)
            if spec is None:
                continue
            text = (get("text", i) or "").strip()
            if not text:
                continue
            section = normalize_section(get("section_number", i))
            key = (spec.key, section, text[:200])
            if key in seen:  # drop duplicate rows
                continue
            seen.add(key)
            status = get("section_status", i) or get("act_status", i) or ""
            parts = _split_words(text, chunk_words, overlap)
            for j, part in enumerate(parts):
                chunks.append(
                    Chunk(
                        chunk_id=f"{spec.key}:{section}:{j}",
                        act_key=spec.key,
                        act_title=str(title),
                        year=int(str(year)[:4]) if year and str(year)[:4].isdigit() else spec.year,
                        chapter=str(get("chapter", i) or ""),
                        section=section,
                        section_title=str(get("section_title", i) or ""),
                        text=part,
                        status=str(status),
                        source_url=str(get("source_url", i) or ""),
                    )
                )
    # Make chunk ids unique even if a section number repeats within an Act (e.g. schedules).
    counts: dict[str, int] = {}
    for c in chunks:
        n = counts.get(c.chunk_id, 0)
        counts[c.chunk_id] = n + 1
        if n:
            c.chunk_id = f"{c.chunk_id}#{n}"
    return chunks


def coverage_report(chunks: list[Chunk]) -> dict[str, int]:
    rep = {spec.key: 0 for spec in ACTS}
    for c in chunks:
        rep[c.act_key] += 1
    return rep
