"""Old criminal codes (IPC / CrPC / Evidence Act) -> 2023 codes (BNS / BNSS / BSA).

The three new codes replaced the old ones from 1 July 2024. Users still ask in old-code
terms ("IPC 420"), so we detect such references, rewrite the query to include the new
provision, and tell the user about the correspondence.

The table in data/old_new_map.csv is a curated subset; verify against the official
Ministry of Home Affairs correspondence tables before relying on it.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache

from .config import DATA_DIR


@dataclass(frozen=True)
class Mapping:
    old_act: str
    old_section: str
    new_act: str
    new_section: str
    topic: str

    def describe(self) -> str:
        return f"{self.old_act} §{self.old_section} → {self.new_act} §{self.new_section} ({self.topic})"


@lru_cache(maxsize=1)
def load_map() -> dict[tuple[str, str], Mapping]:
    out: dict[tuple[str, str], Mapping] = {}
    with open(DATA_DIR / "old_new_map.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = Mapping(r["old_act"], r["old_section"].upper(), r["new_act"], r["new_section"], r["topic"])
            out[(m.old_act, m.old_section)] = m
    return out


# Aliases users type -> canonical act key (old and new codes plus a few common Acts).
ACT_ALIASES = {
    "ipc": "IPC", "indian penal code": "IPC",
    "crpc": "CrPC", "cr.p.c": "CrPC", "cr.p.c.": "CrPC", "code of criminal procedure": "CrPC",
    "iea": "IEA", "evidence act": "IEA", "indian evidence act": "IEA",
    "bns": "BNS", "bharatiya nyaya sanhita": "BNS",
    "bnss": "BNSS", "bharatiya nagarik suraksha sanhita": "BNSS",
    "bsa": "BSA", "bharatiya sakshya adhiniyam": "BSA",
    "cpc": "CPC", "code of civil procedure": "CPC",
    "it act": "IT Act", "information technology act": "IT Act",
    "ni act": "NI Act", "negotiable instruments act": "NI Act",
    "hma": "HMA", "hindu marriage act": "HMA",
    "rti": "RTI", "rti act": "RTI",
    "contract act": "ICA", "indian contract act": "ICA",
    "pocso": "POCSO",
    "constitution": "Constitution",
}
_ALIAS_PATTERN = "|".join(sorted((re.escape(a) for a in ACT_ALIASES), key=len, reverse=True))
_SEC = r"(\d+[A-Za-z]{0,3})"
# "section 302 of IPC", "s. 420 IPC", "sec 65B evidence act", "article 21 of the constitution"
_REF_A = re.compile(
    rf"\b(?:section|sec\.?|s\.|article|art\.?)\s*{_SEC}\s*(?:\(\w+\))*\s*(?:of\s+)?(?:the\s+)?({_ALIAS_PATTERN})\b",
    re.I,
)
# "IPC 302", "IPC section 302", "BNS s. 103"
_REF_B = re.compile(
    rf"\b({_ALIAS_PATTERN})\s*(?:,\s*)?(?:section|sec\.?|s\.|article|art\.?)?\s*{_SEC}\b",
    re.I,
)


@dataclass(frozen=True)
class SectionRef:
    act: str
    section: str


def find_refs(query: str) -> list[SectionRef]:
    refs: list[SectionRef] = []
    for m in _REF_A.finditer(query):
        refs.append(SectionRef(ACT_ALIASES[m.group(2).lower()], m.group(1).upper()))
    for m in _REF_B.finditer(query):
        refs.append(SectionRef(ACT_ALIASES[m.group(1).lower()], m.group(2).upper()))
    # de-duplicate, keep order
    seen, out = set(), []
    for r in refs:
        if (r.act, r.section) not in seen:
            seen.add((r.act, r.section))
            out.append(r)
    return out


def expand_query(query: str) -> tuple[str, list[SectionRef], list[Mapping]]:
    """Return (rewritten query, explicit section refs incl. mapped new ones, mappings used)."""
    refs = find_refs(query)
    table = load_map()
    mappings = [table[(r.act, r.section)] for r in refs if (r.act, r.section) in table]
    extra_refs = [SectionRef(m.new_act, re.sub(r"\(.*", "", m.new_section)) for m in mappings]
    rewritten = query
    if mappings:
        rewritten += " " + " ".join(f"{m.new_act} section {m.new_section} {m.topic}" for m in mappings)
    return rewritten, refs + extra_refs, mappings
