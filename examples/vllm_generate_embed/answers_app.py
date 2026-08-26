# SPDX-License-Identifier: Apache-2.0
"""Browse and semantically search LLM-generated answers stored in LanceDB.

The generation job wrote each prompt, the model's answer, and an embedding of
that answer. This app searches the *answers* by meaning: a query is embedded
with the same model and compared against them, so "how do I make search go
faster" can surface an answer about indexing that never uses those words.

Run with::

    streamlit run examples/vllm_generate_embed/answers_app.py --server.fileWatcherType none -- --uri ./llm_db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from text_embedding import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    embed_text,
    load_embedder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="./llm_db")
    parser.add_argument("--table", default="answers")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    known, _ = parser.parse_known_args()
    return known


@st.cache_resource  # type: ignore[untyped-decorator]
def get_embedder(model_name: str) -> tuple[Any, Any]:
    """Load the embedding model once per session, not once per keystroke."""
    return load_embedder(model_name)


@st.cache_resource  # type: ignore[untyped-decorator]
def get_table(uri: str, table_name: str) -> Any:
    return lancedb.connect(uri).open_table(table_name)


@st.dialog("Row detail", width="large")  # type: ignore[untyped-decorator]
def show_row(record: dict[str, Any]) -> None:
    """Show everything LanceDB stored for one generated answer."""
    st.markdown(f"**Prompt**\n\n{record['prompt']}")
    st.markdown(f"**Response**\n\n{record['response']}")

    scalars = {
        k: str(v) for k, v in record.items() if k not in ("vector", "similarity")
    }
    st.dataframe(
        {"field": list(scalars), "value": list(scalars.values())},
        hide_index=True,
        use_container_width=True,
    )

    vector = np.asarray(record["vector"], np.float32)
    st.markdown(
        f"**Embedding** — {vector.size} dimensions, "
        f"L2 norm {float(np.linalg.norm(vector)):.4f} "
        "(unit length, normalised at write time)"
    )
    with st.expander("Raw embedding values"):
        st.json({"vector": [round(float(v), 5) for v in vector]}, expanded=False)

    if st.button("Find similar answers", use_container_width=True):
        # Search from this row's own embedding: answer-to-answer similarity,
        # no query text involved.
        st.session_state.vector_query = {
            "vector": vector.tolist(),
            "label": record["prompt"][:60],
        }
        st.rerun()


def search_by_vector(table: Any, vector: Any, top_k: int) -> list[dict[str, Any]]:
    """Search, then score the candidates exactly.

    With an IVF_PQ index LanceDB's ``_distance`` is computed against quantised
    vectors: the ranking is right but the value is not true cosine distance.
    Stored vectors are unit length, so a dot product gives the exact score and
    re-scoring ``top_k`` rows costs nothing.
    """
    results: list[dict[str, Any]] = (
        table.search(vector, vector_column_name="vector")
        .metric("cosine")
        .limit(top_k)
        .to_arrow()
        .to_pylist()
    )
    for record in results:
        record["similarity"] = float(np.asarray(record["vector"], np.float32) @ vector)
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="LLM answers", page_icon="💬", layout="wide")
    st.title("Search what the model actually said")

    try:
        table = get_table(args.uri, args.table)
    except Exception as error:  # noqa: BLE001 - surfaced in the UI
        st.error(
            f"Could not open table `{args.table}` at `{args.uri}`.\n\n"
            "Run the generation job first:\n\n"
            "```bash\npython examples/vllm_generate_embed/generate.py "
            f"--engine transformers --uri {args.uri}\n```\n\n({error})"
        )
        return

    num_rows = table.count_rows()
    model, tokenizer = get_embedder(args.embedding_model)

    with st.sidebar:
        st.metric("Answers stored", f"{num_rows:,}")
        top_k = st.slider("Results", min_value=3, max_value=25, value=5)
        st.caption(
            "Answers are matched by meaning, not keywords: the query is "
            "embedded with the same model the answers were."
        )

    vector_query = st.session_state.get("vector_query")

    query = st.text_input(
        "What are you looking for?",
        placeholder="how do I make vector search faster",
        disabled=vector_query is not None,
    )

    if vector_query is not None:
        banner, clear = st.columns([5, 1])
        banner.info(f"Answers similar to: *{vector_query['label']}…*")
        if clear.button("Clear", use_container_width=True):
            del st.session_state.vector_query
            st.rerun()
        started = time.perf_counter()
        results = search_by_vector(
            table, np.asarray(vector_query["vector"], np.float32), top_k
        )
    elif query:
        started = time.perf_counter()
        results = search_by_vector(
            table, embed_text(model, tokenizer, [query])[0], top_k
        )
    else:
        st.info("Type a query above, or browse everything the model generated below.")
        results = table.search(None).limit(top_k).to_arrow().to_pylist()
        for record in results:
            record["similarity"] = float("nan")
        started = time.perf_counter()

    elapsed_ms = (time.perf_counter() - started) * 1000
    if query or vector_query:
        st.caption(f"{len(results)} results in {elapsed_ms:.0f} ms")

    for index, record in enumerate(results):
        with st.container(border=True):
            header, action = st.columns([5, 1])
            similarity = record.get("similarity", float("nan"))
            label = record["prompt"]
            if similarity == similarity:  # not NaN
                label = f"{label}  ·  {similarity:.3f}"
            header.markdown(f"**{label}**")
            if action.button(
                "Inspect", key=f"inspect-{index}", use_container_width=True
            ):
                show_row(record)

            response = str(record["response"])
            preview = response if len(response) <= 400 else response[:400] + "…"
            st.write(preview)
            st.caption(f"{record['model']} · {record['num_chars']} chars")


if __name__ == "__main__":
    main()
