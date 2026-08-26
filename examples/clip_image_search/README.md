# CLIP image search

Embed a directory of JPGs with CLIP across a Ray cluster, write the vectors to
LanceDB, build a vector index, and then search the collection by *describing*
what you want in plain English.

Nothing is labelled, tagged, or filename-parsed. CLIP maps images and text into
one shared embedding space, so a typed sentence and a photograph are directly
comparable — which turns "find the beach photos" into an ordinary vector search.

## Why this example exists

The quickstart proves the library's guarantees on synthetic data. This one shows
the shape of a real workload, where three things matter that a toy example hides:

**The expensive stage is the model, not the IO.** Embedding dominates runtime, so
the pipeline runs CLIP as a Ray *actor* rather than a plain function. Ray keeps
each actor alive across batches, so the model is loaded once per worker instead
of once per batch. With a stateless function you would pay a model load on every
batch and the job would be almost entirely startup cost.

**Reading the files should be distributed too.** `ray.data.read_binary_files`
scans and reads on workers. The driver never holds the images.

**The write should not fragment the table.** Each Ray task commits one
transaction, and a local write lands every fragment in a single atomic version —
so the index is built over a clean table rather than thousands of tiny fragments.

## Setup

```bash
uv pip install -r examples/clip_image_search/requirements.txt
```

The first run downloads the CLIP checkpoint (~600 MB) from Hugging Face.

## 1. Ingest

```bash
python examples/clip_image_search/ingest.py --images ~/Pictures --uri ./demo_db
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--images` | *required* | Directory of JPGs, scanned recursively |
| `--uri` | `./demo_db` | LanceDB directory to write into |
| `--table` | `images` | Table name |
| `--model` | `openai/clip-vit-base-patch32` | Any CLIP checkpoint on the Hub |
| `--device` | `cpu` | Torch device — use `cuda`, or `mps` on Apple silicon |
| `--batch-size` | `32` | Images per forward pass |
| `--concurrency` | `2` | Embedder actors, i.e. parallel model copies |

Tune `--concurrency` to how many model copies fit in memory, and `--batch-size`
to how many images fit in one forward pass. On a GPU, raise both and pass
`--device cuda`.

## 2. Search

```bash
streamlit run examples/clip_image_search/app.py -- --uri ./demo_db
```

The `--` matters: it separates Streamlit's own arguments from the script's.

Type a description and the app embeds it with the *text* half of the same CLIP
model, then runs a vector search against the image embeddings. Results come back
with a cosine similarity score.

## What the pipeline looks like

```
read_binary_files      →  map_batches(ClipEmbedder)  →  write_lancedb  →  create_index
distributed file IO       model loaded once per          one txn per       IVF_PQ over
                          worker, reused per batch       task, atomic      a clean table
```

## Files

| File | Purpose |
| --- | --- |
| `ingest.py` | The Ray pipeline: scan → embed → write → index |
| `app.py` | Streamlit search UI |
| `embedding.py` | Shared CLIP loading and embedding, used by both |
| `requirements.txt` | Extra dependencies |

`embedding.py` is shared deliberately: the image vectors written at ingest time
and the text vectors computed at query time must land in the same space with the
same normalisation, so the model name, dimensionality, and normalisation are
defined once rather than duplicated in two files that can drift apart.

## Notes

**Vectors are L2-normalised at write time.** CLIP similarity is cosine
similarity; normalising once up front makes every stored vector directly
comparable and lets the index work with plain distances.

**The index is skipped on small collections.** IVF_PQ trains IVF centroids and
PQ codebooks with k-means, which needs far more vectors than clusters. Lance
itself warns below 65,536 rows, and PQ is lossy — so on a small collection an
index is slower to build *and less accurate* than the exhaustive scan it
replaces. Under that threshold the script says so and leaves the table
unindexed; a brute-force search over a few thousand 512-dim vectors takes
single-digit milliseconds. Pass `--force-index` to build one anyway and watch
the k-means warnings appear.

**Displayed scores are recomputed, not taken from `_distance`.** This one is
easy to get wrong. Without an index, LanceDB's `_distance` is true cosine
distance and `1 - distance` is the cosine similarity. With an IVF_PQ index the
distance is computed against *quantised* vectors — the ranking it produces is
still correct, but the value is not true cosine distance, and `1 - distance` can
come out negative for a perfectly good match:

| | `_distance` | true cosine | `1 - _distance` |
| --- | ---: | ---: | ---: |
| No index | 0.7156 | 0.2844 | 0.2844 ✓ |
| IVF_PQ | 1.4168 | 0.2916 | −0.4168 ✗ |

Since every stored vector is unit length, the app re-scores the returned
candidates with a dot product against the query — exact cosine similarity, and
free at `top_k` rows. This is the same idea as LanceDB's `refine_factor`:
let the index find candidates fast, then rank them precisely.

**Indexing here calls LanceDB directly.** Distributed index building is not part
of `lancedb-ray` yet — the library covers reads and writes. `table.create_index`
is LanceDB's own API, which is exactly what you would use today.

**Only paths are stored, not image bytes.** The table holds each file's path, so
the app reads originals from disk at display time. If you move or delete the
photos, re-run the ingestion.

**Unreadable files are skipped, not fatal.** A corrupt JPEG logs a warning and
the job continues rather than losing an entire batch.
