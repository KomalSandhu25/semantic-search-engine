"""Streamlit front-end for the Semantic Search Engine.

Provides an interactive UI that communicates with the FastAPI backend
running at ``SEARCH_API_URL`` (default: http://localhost:8000).

Features
--------
* Natural-language query input with configurable ``top_k`` slider.
* Results card for each hit with relevance score badge and keyword
  highlighting so users can quickly spot why a document was returned.
* Latency breakdown bar chart (retrieval vs. reranking) rendered after
  every search.
* "Try these example queries" quick-launch buttons for first-time users.

Run
---
::

    streamlit run streamlit_app.py
    # or, with a non-default API URL:
    SEARCH_API_URL=http://api:8000 streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL: str = os.getenv("SEARCH_API_URL", "http://localhost:8000")
SEARCH_ENDPOINT = f"{API_URL}/search"
HEALTH_ENDPOINT = f"{API_URL}/health"
REQUEST_TIMEOUT_S: int = 30

EXAMPLE_QUERIES: list[str] = [
    "transformer-based sentence embeddings for semantic similarity",
    "dense passage retrieval with FAISS approximate nearest neighbours",
    "cross-encoder reranking for information retrieval",
    "BERT fine-tuning for question answering on SQuAD",
    "contrastive learning for text representation",
    "efficient inference with ONNX and quantisation",
    "retrieval-augmented generation with large language models",
    "multi-label text classification with class imbalance",
]

HIGHLIGHT_COLOUR = "#FFEB3B"  # amber — works on both light and dark themes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _highlight(text: str, query: str) -> str:
    """Return *text* with query terms wrapped in HTML ``<mark>`` tags.

    The match is case-insensitive.  Each unique non-stop word in *query*
    longer than two characters is highlighted independently.

    Args:
        text: The passage text to annotate.
        query: The raw query string whose terms are highlighted.

    Returns:
        An HTML string safe for ``st.markdown(..., unsafe_allow_html=True)``.

    Example::

        _highlight("Dense retrieval is fast.", "dense retrieval")
        # -> highlighted HTML string
    """
    STOP_WORDS = {
        "a", "an", "the", "is", "in", "on", "at", "to", "for", "of",
        "and", "or", "but", "with", "from", "by", "as", "are", "was",
        "it", "its", "be", "been",
    }
    terms = [
        w for w in re.split(r"\W+", query.lower())
        if len(w) > 2 and w not in STOP_WORDS
    ]

    mark_style = f"background-color:{HIGHLIGHT_COLOUR};border-radius:2px;padding:0 2px;"

    result = text
    for term in sorted(set(terms), key=len, reverse=True):
        result = re.sub(
            f"({re.escape(term)})",
            f'<mark style="{mark_style}">\\1</mark>',
            result,
            flags=re.IGNORECASE,
        )
    return result


def _api_healthy() -> bool:
    """Return *True* if the backend health endpoint responds with 200 OK."""
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _do_search(query: str, top_k: int) -> dict[str, Any] | None:
    """Call the ``/search`` endpoint and return the parsed JSON payload.

    Args:
        query: User search query.
        top_k: Maximum results to request from the API.

    Returns:
        Parsed response dict on success, or *None* on error (error already
        surfaced via ``st.error``).
    """
    try:
        resp = requests.post(
            SEARCH_ENDPOINT,
            json={"query": query, "top_k": top_k},
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Cannot reach the search API at **{API_URL}**. "
            "Is the API container running?  Check `docker-compose up`."
        )
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Request timed out after {REQUEST_TIMEOUT_S} s.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    return None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="\U0001f50d",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Search Settings")

    top_k: int = st.slider(
        "Results to return (top_k)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help=(
            "How many re-ranked results to display.  The API always retrieves "
            "4x this number in Stage 1 before the cross-encoder re-ranks them."
        ),
    )

    st.divider()

    # API health indicator
    st.subheader("API Status")
    if _api_healthy():
        st.success(f"✅ Connected — {API_URL}")
    else:
        st.warning(f"⚠️ Cannot reach — {API_URL}")

    st.divider()
    st.caption(
        "**Architecture**: Bi-Encoder (all-MiniLM-L6-v2) → FAISS IVF → "
        "Cross-Encoder (ms-marco-MiniLM-L-6-v2).  "
        "Corpus: MS MARCO passages."
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("\U0001f50d Semantic Search Engine")
st.markdown(
    "Two-stage retrieval: **bi-encoder + FAISS** for recall, "
    "**cross-encoder** for precision.  Enter a query below."
)

# Query input
query_input: str = st.text_input(
    "Search query",
    placeholder="e.g. transformer-based sentence embeddings",
    label_visibility="collapsed",
)

search_clicked = st.button("Search", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

st.markdown("**Try these example queries:**")
cols = st.columns(4)
for i, example in enumerate(EXAMPLE_QUERIES):
    short = textwrap.shorten(example, width=42, placeholder="…")
    if cols[i % 4].button(short, key=f"ex_{i}", use_container_width=True):
        query_input = example
        search_clicked = True

st.divider()

# ---------------------------------------------------------------------------
# Execute search
# ---------------------------------------------------------------------------

if search_clicked and query_input.strip():
    query = query_input.strip()

    with st.spinner(f"Searching for *{query}* …"):
        data = _do_search(query, top_k)

    if data is not None:
        results: list[dict] = data.get("results", [])
        query_time_ms: float = data.get("query_time_ms", 0.0)
        stages: dict = data.get("stages", {})
        retrieval_ms: float = stages.get("retrieval_ms", 0.0)
        reranking_ms: float = stages.get("reranking_ms", 0.0)

        # ------------------------------------------------------------------
        # Latency breakdown chart
        # ------------------------------------------------------------------

        col_metrics, col_chart = st.columns([1, 2])

        with col_metrics:
            st.metric("Total latency", f"{query_time_ms:.1f} ms")
            st.metric("Results returned", len(results))
            st.metric("Retrieval (Stage 1)", f"{retrieval_ms:.1f} ms")
            st.metric("Reranking (Stage 2)", f"{reranking_ms:.1f} ms")

        with col_chart:
            latency_df = pd.DataFrame(
                {
                    "Stage": ["Bi-Encoder + FAISS (Stage 1)", "Cross-Encoder (Stage 2)"],
                    "Latency (ms)": [retrieval_ms, reranking_ms],
                }
            ).set_index("Stage")
            st.bar_chart(
                latency_df,
                y="Latency (ms)",
                color="#4C8BF5",
                height=220,
                use_container_width=True,
            )

        st.divider()

        # ------------------------------------------------------------------
        # Result cards
        # ------------------------------------------------------------------

        if not results:
            st.info("No results returned.  Try a different query or rebuild the index.")
        else:
            st.subheader(f"Top {len(results)} results for: *{query}*")

            for rank, hit in enumerate(results, start=1):
                score: float = hit.get("score", 0.0)
                text: str = hit.get("text", "")
                doc_id: str = str(hit.get("doc_id", ""))
                metadata: dict = hit.get("metadata", {})

                # Score badge colour: green >= 0.8, orange >= 0.5, red < 0.5
                if score >= 0.8:
                    badge_colour = "#2ECC71"
                elif score >= 0.5:
                    badge_colour = "#F39C12"
                else:
                    badge_colour = "#E74C3C"

                badge_html = (
                    f'<span style="background:{badge_colour};color:white;'
                    f'padding:2px 8px;border-radius:12px;font-size:0.8em;"'
                    f'>{score:.3f}</span>'
                )

                with st.container(border=True):
                    header_cols = st.columns([0.05, 0.7, 0.25])
                    header_cols[0].markdown(f"**#{rank}**")
                    header_cols[1].markdown(f"doc_id: `{doc_id}`")
                    header_cols[2].markdown(badge_html, unsafe_allow_html=True)

                    highlighted_text = _highlight(text, query)
                    st.markdown(highlighted_text, unsafe_allow_html=True)

                    if metadata:
                        with st.expander("Metadata", expanded=False):
                            st.json(metadata)

elif search_clicked and not query_input.strip():
    st.warning("Please enter a search query.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Semantic Search Engine · "
    "[GitHub](https://github.com/KomalSandhu25/semantic-search-engine) · "
    "Built with sentence-transformers, FAISS, FastAPI, and Streamlit"
)
