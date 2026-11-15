from praetor.config import Settings
from praetor.data import load_chunks, match_act, normalize_section
from praetor.generate import build_messages, check_citations, should_refuse, strip_thinking
from praetor.index import Index, build_index
from praetor.mapping import expand_query, find_refs
from praetor.retrieve import Retriever, rrf


# ---- data -------------------------------------------------------------------
def test_match_act_years_and_amendments():
    assert match_act("The Code of Criminal Procedure, 1973", 1974).key == "CrPC"  # off-by-one year
    assert match_act("The Indian Penal Code (Amendment) Act, 2013", 2013) is None
    assert match_act("The Companies Act, 1956", 1956) is None  # only the 2013 Act is configured
    assert match_act("The Factories Act, 1948", 1948) is None


def test_normalize_section():
    assert normalize_section("Section 304B.") == "304B"
    assert normalize_section("Article 21") == "21"
    assert normalize_section(138) == "138"


def test_load_chunks_filters_and_chunks(parquet_path):
    chunks = load_chunks(parquet_path, chunk_words=100, overlap=20)
    acts = {c.act_key for c in chunks}
    assert acts == {"BNS", "IPC", "CrPC", "NI Act", "Constitution", "ICA"}
    contract = [c for c in chunks if c.act_key == "ICA"]
    assert len(contract) > 1                      # long section was split
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    art21 = next(c for c in chunks if c.act_key == "Constitution")
    assert art21.section == "21" and art21.label == "Constitution Article 21"


# ---- mapping ----------------------------------------------------------------
def test_find_refs_formats():
    refs = {(r.act, r.section) for r in find_refs("Is section 302 of IPC same as BNS 103? also s. 65B evidence act")}
    assert {("IPC", "302"), ("BNS", "103"), ("IEA", "65B")} <= refs


def test_expand_query_maps_old_to_new():
    rewritten, refs, maps = expand_query("What is IPC 420 now?")
    assert maps and maps[0].new_act == "BNS" and maps[0].new_section == "318"
    assert ("BNS", "318") in {(r.act, r.section) for r in refs}
    assert "BNS section 318" in rewritten


# ---- retrieval --------------------------------------------------------------
def make_retriever(parquet_path, embedder, reranker, tmp_path):
    idx = build_index(load_chunks(parquet_path), embedder)
    idx.save(tmp_path / "idx")
    idx = Index.load(tmp_path / "idx")             # exercise persistence
    return Retriever(idx, embedder, reranker, Settings())


def test_rrf_prefers_items_in_both_lists():
    s = rrf([[1, 2, 3], [3, 1, 4]])
    assert max(s, key=s.get) in (1, 3) and s[1] > s[2]


def test_search_modes(parquet_path, embedder, reranker, tmp_path):
    r = make_retriever(parquet_path, embedder, reranker, tmp_path)
    for mode in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        res = r.search("cheque returned unpaid insufficient funds", mode=mode, top_k=3)
        assert res.hits[0].chunk.act_key == "NI Act", mode


def test_old_code_question_surfaces_new_section(parquet_path, embedder, reranker, tmp_path):
    r = make_retriever(parquet_path, embedder, reranker, tmp_path)
    res = r.search("What replaced IPC 420?", top_k=3)
    top = {(h.chunk.act_key, h.chunk.section) for h in res.hits[:2]}
    assert ("BNS", "318") in top and ("IPC", "420") in top


def test_act_filter(parquet_path, embedder, reranker, tmp_path):
    r = make_retriever(parquet_path, embedder, reranker, tmp_path)
    res = r.search("punishment for murder", acts=["IPC"], top_k=5)
    assert res.hits and all(h.chunk.act_key == "IPC" for h in res.hits)


def test_refusal_gate(parquet_path, embedder, reranker, tmp_path):
    r = make_retriever(parquet_path, embedder, reranker, tmp_path)
    res = r.search("who won the cricket world cup", top_k=3)
    assert should_refuse(res, Settings(), reranked=True)
    res = r.search("punishment for murder", top_k=3)
    assert not should_refuse(res, Settings(), reranked=True)


# ---- generation helpers -----------------------------------------------------
def test_prompt_contains_sources_and_mapping(parquet_path, embedder, reranker, tmp_path):
    r = make_retriever(parquet_path, embedder, reranker, tmp_path)
    res = r.search("IPC 302 punishment", top_k=3)
    msgs = build_messages("IPC 302 punishment", res)
    assert "[S1]" in msgs[1]["content"] and "IPC §302 → BNS §103" in msgs[1]["content"]


def test_strip_thinking():
    assert strip_thinking("<think>hmm</think>\nAnswer [S1].") == "Answer [S1]."
    assert strip_thinking("<think>still thinking") == ""


def test_check_citations():
    rep = check_citations("Murder is punished with death or life imprisonment [S1]. "
                          "This sentence has no citation at all here. Bad ref here too right [S9].", 3)
    assert rep.cited == [1] and rep.invalid == [9] and rep.uncited_sentences == 1
