# SPDX-License-Identifier: Apache-2.0
"""Streamlit app for searching an embedded image collection in plain English.

The point of the example: once the ingestion job has embedded a directory of
photos, you can find images by *describing* them. Nothing was labelled, no
filenames were parsed, no tags were written. CLIP puts images and text in one
embedding space, so a typed sentence and a photograph are directly comparable
and the query is an ordinary vector search.

Run with::

    streamlit run examples/clip_image_search/app.py --server.fileWatcherType none -- --uri ./demo_db
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


def image_source(record: dict[str, Any]) -> Any:
    """Return something ``st.image`` can render for this row.

    The JPEG bytes are stored in the table, so results render straight from
    LanceDB and keep working after the originals are moved or deleted. Tables
    written before the column existed still carry a path, so fall back to it.
    """
    payload = record.get("image")
    if payload:
        return bytes(payload)
    path = record.get("path")
    return path if path and Path(path).exists() else None


def search_by_vector(table: Any, vector: Any, top_k: int) -> list[dict[str, Any]]:
    """Run a vector search and score the candidates exactly.

    LanceDB's ``_distance`` is computed against quantised vectors when an
    IVF_PQ index is present: the ranking is right, but the value is not true
    cosine distance and ``1 - distance`` can come out negative. Every stored
    vector is unit length, so a dot product against the query gives the exact
    cosine similarity, and re-scoring ``top_k`` rows costs nothing.
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


@st.dialog("Row detail", width="large")  # type: ignore[untyped-decorator]
def show_row(record: dict[str, Any]) -> None:
    """Render everything LanceDB stored for one row.

    The point of the example is that the table is ordinary data, not a black
    box: the same row that backs a search hit can be inspected field by field,
    embedding included.
    """
    left, right = st.columns([2, 3])

    with left:
        source = image_source(record)
        if source is not None:
            st.image(source, use_container_width=True)
        else:
            st.warning("No image data for this row")

    with right:
        st.metric("Cosine similarity", f"{record['similarity']:.4f}")
        # Summarise the stored bytes rather than dumping them into the table,
        # and keep the 512-dim embedding out of it entirely.
        scalars: dict[str, str] = {}
        for key, value in record.items():
            if key in ("vector", "similarity"):
                continue
            if isinstance(value, (bytes, bytearray)):
                scalars[key] = f"<{len(value):,} bytes of JPEG, stored in the table>"
            else:
                scalars[key] = str(value)
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

    if st.button("Find similar images", use_container_width=True):
        # Reuse the row's own embedding as the query: image-to-image search
        # with no text involved.
        st.session_state.image_query = {
            "vector": vector.tolist(),
            "label": record["filename"],
        }
        st.rerun()


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

    image_query = st.session_state.get("image_query")

    query = st.text_input(
        "Describe what you are looking for",
        placeholder="a dog running on a beach at sunset",
        disabled=image_query is not None,
    )
    st.caption("Try: " + " · ".join(f"`{e}`" for e in EXAMPLE_QUERIES))

    if image_query is not None:
        banner, clear = st.columns([5, 1])
        banner.info(f"Showing images similar to **{image_query['label']}**")
        if clear.button("Clear", use_container_width=True):
            del st.session_state.image_query
            st.rerun()
        started = time.perf_counter()
        results = search_by_vector(
            table, np.asarray(image_query["vector"], np.float32), top_k
        )
    elif query:
        started = time.perf_counter()
        results = search_by_vector(
            table, embed_text(model, processor, [query])[0], top_k
        )
    else:
        st.info(
            "Type a description above to search, then click any result to "
            "inspect the row behind it."
        )
        return

    elapsed_ms = (time.perf_counter() - started) * 1000
    st.caption(f"{len(results)} results in {elapsed_ms:.0f} ms")

    if not results:
        st.warning("No results.")
        return

    columns_per_row = 4
    for start in range(0, len(results), columns_per_row):
        chunk = results[start : start + columns_per_row]
        for column, record in zip(st.columns(columns_per_row), chunk, strict=False):
            with column:
                source = image_source(record)
                if source is not None:
                    st.image(source, use_container_width=True)
                else:
                    st.warning(f"No image data:\n`{record.get('path', '?')}`")

                caption = record["filename"]
                if show_scores:
                    caption = f"{caption} — {record['similarity']:.3f}"
                # Streamlit cannot attach a click handler to an image, so the
                # button beneath it is what opens the row.
                if st.button(
                    caption,
                    key=f"inspect-{start}-{record['filename']}-{record['path']}",
                    use_container_width=True,
                ):
                    show_row(record)


if __name__ == "__main__":
    main()
