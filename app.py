"""Praetor: Streamlit front end."""
from __future__ import annotations

import os
import re
from html import escape

import streamlit as st

from praetor.config import ACTS, INDEX_DIR, ROOT, Settings
from praetor.generate import (DISCLAIMER, LLM, REFUSAL, build_messages, check_citations,
                              should_refuse)
from praetor.index import FastEmbedder, FastReranker, Index
from praetor.retrieve import Retriever

st.set_page_config(page_title="Praetor · Indian Law Q&A", page_icon="⚖️", layout="wide")

# Streamlit Cloud secrets -> env, so Settings picks them up.
for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:  # no secrets file locally
        pass

st.markdown(
    """
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.praetor-title {font-size: 2.1rem; font-weight: 700; margin-bottom: 0;}
.praetor-sub {color: #6b7280; margin-top: .2rem;}
.src-card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .7rem .9rem; margin-bottom: .6rem;}
.src-head {font-weight: 600;}
.src-meta {color: #6b7280; font-size: .82rem;}
.src-body {font-size: .9rem; white-space: pre-wrap; margin-top: .4rem; max-height: 220px; overflow-y: auto;}
.badge {display:inline-block; padding:1px 8px; border-radius: 999px; font-size:.75rem; margin-left:.4rem;
        background: rgba(37,99,235,.12); color:#2563eb;}
.badge.warn {background: rgba(220,38,38,.12); color:#dc2626;}
.map-note {border-left: 3px solid #2563eb; padding: .4rem .8rem; background: rgba(37,99,235,.06);
           border-radius: 4px; margin-bottom: .6rem; font-size: .9rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_embedder():
    return FastEmbedder()


@st.cache_resource(show_spinner=False)
def get_reranker():
    return FastReranker()


@st.cache_resource(show_spinner=False)
def get_index() -> Index:
    if Index.exists(INDEX_DIR):
        return Index.load(INDEX_DIR)
    # No prebuilt index committed: build once on this machine and cache it.
    from praetor.build import build_from_parquet, download_central

    with st.status("First run: building the index from Open India Law…", expanded=True) as status:
        parquet = download_central(ROOT / ".cache" / "in_central_legislation.parquet")
        index = build_from_parquet(parquet, get_embedder(), Settings(), progress=st.write)
        index.save(INDEX_DIR)
        status.update(label="Index ready", state="complete", expanded=False)
    return index


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever(get_index(), get_embedder(), get_reranker(), Settings())


# ---------------------------------------------------------------- helpers
def render_answer(text: str, n: int) -> str:
    """Turn [S3] into a superscript-style marker."""
    return re.sub(r"\[S(\d+)\]", lambda m: f"**[{m.group(1)}]**" if int(m.group(1)) <= n else "", text)


def source_card(i: int, hit) -> str:
    c = hit.chunk
    warn = c.status and c.status.lower() not in ("in_force", "")
    badge = f'<span class="badge warn">{escape(c.status)}</span>' if warn else ""
    if c.act_key in ("IPC", "CrPC", "IEA"):
        badge += '<span class="badge warn">replaced 1 Jul 2024</span>'
    link = f' · <a href="{escape(c.source_url)}" target="_blank">source</a>' if c.source_url.startswith("http") else ""
    return (
        f'<div class="src-card"><div class="src-head">[{i}] {escape(c.label)} — {escape(c.section_title)}{badge}</div>'
        f'<div class="src-meta">{escape(c.act_title)} · relevance {hit.score:.2f}{link}</div>'
        f'<div class="src-body">{escape(c.text)}</div></div>'
    )


# ---------------------------------------------------------------- sidebar
settings = Settings()
try:
    meta = get_index().meta
except Exception as e:  # noqa: BLE001
    st.error(f"Index failed to load: {e}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    act_keys = [a.key for a in ACTS]
    picked = st.multiselect("Limit to Acts", act_keys, placeholder="All indexed Acts")
    mode = st.selectbox("Retrieval", ["hybrid+rerank", "hybrid", "dense", "bm25"],
                        help="hybrid = BM25 + dense fused with reciprocal rank fusion; +rerank adds a cross-encoder.")
    settings.top_k = st.slider("Sources passed to the model", 3, 10, settings.top_k)
    show_debug = st.toggle("Show retrieval details", value=False)
    st.divider()
    llm_ready = bool(settings.llm_api_key) or "localhost" in settings.llm_base_url
    st.caption(f"**Model:** `{settings.llm_model}`" if llm_ready else "**Model:** not configured, so Praetor shows sources only")
    st.caption(f"**Index:** {meta['n_chunks']:,} passages · {meta.get('dataset', '')}")
    st.caption("Data: [Open India Law](https://huggingface.co/datasets/vaquill/open-india-law) by Vaquill AI, CC BY 4.0.")

# ---------------------------------------------------------------- header
st.markdown('<p class="praetor-title">⚖️ Praetor</p>', unsafe_allow_html=True)
st.markdown('<p class="praetor-sub">Ask questions about Indian central law. Every answer cites the section it comes from.</p>',
            unsafe_allow_html=True)

EXAMPLES = [
    "What replaced IPC 420 under the new criminal law?",
    "Can a wife claim maintenance from her husband?",
    "Are electronic records admissible as evidence?",
    "What happens when a cheque bounces?",
]
if "history" not in st.session_state:
    st.session_state.history = []

cols = st.columns(len(EXAMPLES))
clicked = None
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        clicked = ex

# replay history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown(turn["answer_md"], unsafe_allow_html=True)
        with st.expander(f"Sources ({len(turn['cards'])})"):
            st.markdown("".join(turn["cards"]), unsafe_allow_html=True)

question = st.chat_input("e.g. What is the punishment for dowry death?") or clicked
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Searching the Acts…"):
            retriever = get_retriever()
            result = retriever.search(question, acts=picked or None, mode=mode, top_k=settings.top_k)

        notes = ""
        if result.mappings:
            items = "<br>".join(escape(m.describe()) for m in result.mappings)
            notes = f'<div class="map-note"><b>Old → new code</b><br>{items}</div>'
            st.markdown(notes, unsafe_allow_html=True)

        cards = [source_card(i, h) for i, h in enumerate(result.hits, 1)]
        refuse = should_refuse(result, settings, reranked=(mode == "hybrid+rerank"))
        answer = ""
        if refuse:
            answer = REFUSAL + " Try rephrasing, or remove the Act filter."
            st.markdown(answer)
        elif not llm_ready:
            answer = "Answer generation is off because no LLM key is configured. Here are the most relevant provisions."
            st.info(answer)
        else:
            box = st.empty()
            try:
                for partial in LLM(settings).stream(build_messages(question, result)):
                    answer = partial
                    box.markdown(render_answer(answer, len(result.hits)) + " ▌")
                box.markdown(render_answer(answer, len(result.hits)))
            except Exception as e:  # noqa: BLE001
                answer = f"The language model call failed ({type(e).__name__}). The retrieved sources are below."
                box.warning(answer)
            else:
                rep = check_citations(answer, len(result.hits))
                if rep.invalid or (rep.total_sentences and rep.coverage < 0.6):
                    st.caption(f"⚠️ Citation check: {rep.coverage:.0%} of sentences cite a source"
                               + (f"; invalid refs {rep.invalid}" if rep.invalid else ""))
        st.caption(DISCLAIMER)

        with st.expander(f"Sources ({len(cards)})", expanded=refuse or not llm_ready):
            st.markdown("".join(cards), unsafe_allow_html=True)

        if show_debug:
            with st.expander("Retrieval details"):
                st.write({"rewritten_query": result.rewritten,
                          "explicit_refs": [f"{r.act} {r.section}" for r in result.refs],
                          "best_score": round(result.best_score, 3)})
                st.dataframe([{"rank": i, "source": h.chunk.label, "score": round(h.score, 3), **h.ranks}
                              for i, h in enumerate(result.hits, 1)], use_container_width=True)

    st.session_state.history.append({
        "q": question,
        "answer_md": notes + render_answer(answer, len(result.hits)) + f"\n\n<small>{DISCLAIMER}</small>",
        "cards": cards,
    })
