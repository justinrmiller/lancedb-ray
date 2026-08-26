# SPDX-License-Identifier: Apache-2.0
"""Streamlit app for searching an embedded image collection in plain English.

The point of the example: once the ingestion job has embedded a directory of
photos, you can find images by *describing* them. Nothing was labelled, no
filenames were parsed, no tags were written. CLIP puts images and text in one
embedding space, so a typed sentence and a photograph are directly comparable
and the query is an ordinary vector search.

Run with::

    streamlit run examples/clip_image_search/app.py -- --uri ./demo_db
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

from embedding import DEFAULT_MODEL, embed_text, load_clip  # noqa: E402

#: Shown under the search box so the app is usable without knowing the corpus.
EXAMPLE_QUERIES = (
    "something red",
    "a person smiling",
    "a city street at night",
    "food on a plate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="./demo_db")
    parser.add_argument("--table", default="images")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    # Streamlit passes the script's own argv through after `--`.
    known, _ = parser.parse_known_args()
    return known


@st.cache_resource  # type: ignore[untyped-decorator]
def get_model(model_name: str) -> tuple[Any, Any]:
    """Load CLIP once per Streamlit session, not once per keystroke."""
    return load_clip(model_name)


@st.cache_resource  # type: ignore[untyped-decorator]
def get_table(uri: str, table_name: str) -> Any:
    return lancedb.connect(uri).open_table(table_name)


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="CLIP image search", page_icon="🔎", layout="wide")
    st.title("Search your photos by describing them")

    try:
        table = get_table(args.uri, args.table)
    except Exception as error:  # noqa: BLE001 - surfaced in the UI
        st.error(
            f"Could not open table `{args.table}` at `{args.uri}`.\n\n"
            f"Run the ingestion job first:\n\n"
            f"```bash\npython examples/clip_image_search/ingest.py "
            f"--images <your-photos> --uri {args.uri}\n```\n\n"
            f"({error})"
        )
        return

    num_rows = table.count_rows()
    model, processor = get_model(args.model)

    with st.sidebar:
        st.metric("Images indexed", f"{num_rows:,}")
        top_k = st.slider("Results", min_value=3, max_value=48, value=12, step=3)
        show_scores = st.checkbox("Show similarity scores", value=True)
        st.caption(
            "These images were never labelled or tagged. Matching happens "
            "entirely in CLIP's shared image/text embedding space."
        )

    query = st.text_input(
        "Describe what you are looking for",
        placeholder="a dog running on a beach at sunset",
    )

    st.caption("Try: " + " · ".join(f"`{e}`" for e in EXAMPLE_QUERIES))

    if not query:
        st.info("Type a description above to search.")
        return

    started = time.perf_counter()
    vector = embed_text(model, processor, [query])[0]
    results = (
        table.search(vector, vector_column_name="vector")
        .metric("cosine")
        .limit(top_k)
        .to_arrow()
        .to_pylist()
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    # Score the returned candidates exactly rather than reporting LanceDB's
    # ``_distance``. With an IVF_PQ index that distance is computed against
    # quantised vectors: the ranking it produces is right, but the value is not
    # true cosine distance and ``1 - distance`` can even come out negative.
    # Every stored vector is unit length, so a dot product against the query is
    # the exact cosine similarity -- and re-scoring top_k rows costs nothing.
    for record in results:
        record["similarity"] = float(np.asarray(record["vector"], np.float32) @ vector)
    results.sort(key=lambda r: r["similarity"], reverse=True)

    st.caption(f"{len(results)} results in {elapsed_ms:.0f} ms")

    if not results:
        st.warning("No results.")
        return

    columns_per_row = 4
    for start in range(0, len(results), columns_per_row):
        row = results[start : start + columns_per_row]
        for column, record in zip(st.columns(columns_per_row), row, strict=False):
            with column:
                path = record["path"]
                if Path(path).exists():
                    st.image(path, use_container_width=True)
                else:
                    st.warning(f"Missing file:\n`{path}`")
                caption = record["filename"]
                if show_scores:
                    caption = f"{caption} — {record['similarity']:.3f}"
                st.caption(caption)


if __name__ == "__main__":
    main()
