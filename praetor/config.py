"""Central configuration for Praetor."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(os.getenv("PRAETOR_INDEX_DIR", ROOT / "index"))
DATA_DIR = ROOT / "data"

DATASET_VERSION = "v2026.08.1"
CENTRAL_PARQUET_URL = (
    f"https://oss-data-in.vaquill.ai/{DATASET_VERSION}/in_central_legislation.parquet"
)
HF_DATASET = "vaquill/open-india-law"
HF_CENTRAL_FILE = "in_central_legislation.parquet"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class ActSpec:
    """An Act to index. Matched on a case-insensitive title substring plus year."""

    key: str          # short code used in the UI and in citations, e.g. "BNS"
    title: str        # substring expected in the dataset title
    year: int | None  # enactment year (None = don't check)


# v1 corpus: the core central Acts people actually ask about.
ACTS: list[ActSpec] = [
    ActSpec("Constitution", "Constitution of India", None),
    ActSpec("BNS", "Bharatiya Nyaya Sanhita", 2023),
    ActSpec("BNSS", "Bharatiya Nagarik Suraksha Sanhita", 2023),
    ActSpec("BSA", "Bharatiya Sakshya Adhiniyam", 2023),
    ActSpec("IPC", "Indian Penal Code", 1860),
    ActSpec("CrPC", "Code of Criminal Procedure", 1973),
    ActSpec("IEA", "Indian Evidence Act", 1872),
    ActSpec("CPC", "Code of Civil Procedure", 1908),
    ActSpec("ICA", "Indian Contract Act", 1872),
    ActSpec("SRA", "Specific Relief Act", 1963),
    ActSpec("TPA", "Transfer of Property Act", 1882),
    ActSpec("Limitation", "Limitation Act", 1963),
    ActSpec("NI Act", "Negotiable Instruments Act", 1881),
    ActSpec("Arbitration", "Arbitration and Conciliation Act", 1996),
    ActSpec("CPA", "Consumer Protection Act", 2019),
    ActSpec("IT Act", "Information Technology Act", 2000),
    ActSpec("DPDP", "Digital Personal Data Protection Act", 2023),
    ActSpec("RTI", "Right to Information Act", 2005),
    ActSpec("HMA", "Hindu Marriage Act", 1955),
    ActSpec("HSA", "Hindu Succession Act", 1956),
    ActSpec("SMA", "Special Marriage Act", 1954),
    ActSpec("DV Act", "Protection of Women from Domestic Violence Act", 2005),
    ActSpec("Dowry", "Dowry Prohibition Act", 1961),
    ActSpec("POCSO", "Protection of Children from Sexual Offences Act", 2012),
    ActSpec("JJ Act", "Juvenile Justice (Care and Protection of Children) Act", 2015),
    ActSpec("NDPS", "Narcotic Drugs and Psychotropic Substances Act", 1985),
    ActSpec("MV Act", "Motor Vehicles Act", 1988),
    ActSpec("Companies", "Companies Act", 2013),
]


@dataclass
class Settings:
    # Retrieval
    bm25_k: int = 50
    dense_k: int = 50
    fuse_k: int = 30
    top_k: int = 6
    rrf_k: int = 60
    # Evidence gate: below this best rerank score we refuse instead of answering.
    min_rerank_score: float = -2.0
    # Chunking (in words)
    chunk_words: int = 320
    chunk_overlap: int = 60
    # LLM (any OpenAI-compatible endpoint serving a Qwen model)
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen/qwen3-32b"))
    temperature: float = 0.1
    max_tokens: int = 900
