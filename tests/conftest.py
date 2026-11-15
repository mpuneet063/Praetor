"""Synthetic fixture in the Open India Law legislation schema plus offline fake models."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROWS = [
    # act_id, title, year, chapter, section_number, section_title, text, act_status
    ("IND_central_2023_45", "The Bharatiya Nyaya Sanhita, 2023", 2023, "Chapter VI", "103",
     "Punishment for murder", "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.", "in_force"),
    ("IND_central_2023_45", "The Bharatiya Nyaya Sanhita, 2023", 2023, "Chapter XVII", "318",
     "Cheating", "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property shall be punished.", "in_force"),
    ("IND_central_2023_45", "The Bharatiya Nyaya Sanhita, 2023", 2023, "Chapter V", "80",
     "Dowry death", "Where the death of a woman is caused by any burns or bodily injury within seven years of her marriage in connection with demand for dowry, such death shall be called dowry death.", "in_force"),
    ("IND_central_1860_45", "The Indian Penal Code, 1860", 1860, "Chapter XVI", "302",
     "Punishment for murder", "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.", "repealed"),
    ("IND_central_1860_45", "The Indian Penal Code, 1860", 1860, "Chapter XVII", "420",
     "Cheating and dishonestly inducing delivery of property", "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property shall be punished with imprisonment up to seven years.", "repealed"),
    ("IND_central_1974_2", "The Code of Criminal Procedure, 1973", 1974, "Chapter IX", "125",
     "Order for maintenance of wives, children and parents", "If any person having sufficient means neglects or refuses to maintain his wife, unable to maintain herself, a Magistrate may order a monthly allowance.", "repealed"),
    ("IND_central_1881_26", "The Negotiable Instruments Act, 1881", 1881, "Chapter XVII", "138",
     "Dishonour of cheque for insufficiency of funds", "Where any cheque drawn by a person is returned by the bank unpaid because the amount of money standing to the credit of that account is insufficient, such person shall be deemed to have committed an offence.", "in_force"),
    ("IND_central_1950_0", "Constitution of India", 1950, "Part III", "Article 21",
     "Protection of life and personal liberty", "No person shall be deprived of his life or personal liberty except according to procedure established by law.", "in_force"),
    # noise: not in the configured Act list, and an amendment Act
    ("IND_central_1948_11", "The Factories Act, 1948", 1948, "Chapter I", "2",
     "Interpretation", "In this Act, unless there is anything repugnant, adult means a person who has completed eighteen years.", "in_force"),
    ("IND_central_2013_13", "The Indian Penal Code (Amendment) Act, 2013", 2013, "", "1",
     "Short title", "This Act may be called the Indian Penal Code (Amendment) Act.", "in_force"),
    # long section, to exercise chunking
    ("IND_central_1872_9", "The Indian Contract Act, 1872", 1872, "Chapter II", "10",
     "What agreements are contracts", " ".join(["All agreements are contracts if they are made by the free consent of parties competent to contract."] * 40), "in_force"),
]


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    cols = list(zip(*ROWS))
    table = pa.table({
        "act_id": cols[0], "title": cols[1], "year": cols[2], "chapter": cols[3],
        "section_number": cols[4], "section_title": cols[5], "text": cols[6], "act_status": cols[7],
        "source_url": ["https://www.indiacode.nic.in/x"] * len(ROWS),
    })
    p = tmp_path / "central.parquet"
    pq.write_table(table, p)
    return p


def _tokens(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


class FakeEmbedder:
    """Hashed bag-of-words embeddings, deterministic and offline."""

    dim = 256

    def _vec(self, t: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(t):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        return v / max(np.linalg.norm(v), 1e-9)

    def embed_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    def score(self, query, texts):
        content = lambda s: {w for w in _tokens(s) if len(w) > 3}  # noqa: E731  crude stopword filter
        q = content(query)
        return np.array([len(q & content(t)) - 2.5 for t in texts], dtype=np.float32)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def reranker():
    return FakeReranker()
