# Generate with an LLM, embed, and search

Answer a list of prompts with an LLM, embed each answer, write prompt + answer +
vector to LanceDB in one atomic commit, then search the answers *by meaning*
from a Streamlit app.

The point is the shape of the pipeline: generation and embedding are both
expensive model steps that Ray fans out across a cluster, and the result lands
in a table you can query — rather than a pile of JSON someone has to post-process.

## The pipeline

```
prompts  →  generate  →  embed  →  write_lancedb
            vLLM engines   sentence model    one atomic
            across Ray     per worker        commit
```

Both model stages are Ray actors, not plain functions, so each worker loads its
model once and reuses it across batches. With a stateless function you would pay
a model load per batch and the job would be mostly startup.

## Two generation backends

| `--engine` | What it runs | Needs |
| --- | --- | --- |
| `vllm` (default) | Ray Data's [LLM processor](https://docs.ray.io/en/latest/data/working-with-llms.html) driving vLLM engines | CUDA GPUs |
| `transformers` | A small chat model locally | Nothing beyond torch |

They emit the same columns, so everything downstream is identical. The
`transformers` path exists so the pipeline can be exercised end to end on a
laptop — vLLM does not install usefully on Apple silicon, and a GPU-only example
is one nobody can try before committing to a cluster.

**The `vllm` path is untested here.** It is written against Ray 2.58's
`build_processor` / `vLLMEngineProcessorConfig` API, verified against the
installed package rather than the docs alone, but this machine has no CUDA GPU.
The `transformers` path is what was actually run.

## Setup

```bash
uv pip install -r examples/vllm_generate_embed/requirements.txt
```

For the vLLM path, additionally:

```bash
uv pip install "ray[llm]" vllm
```

## 1. Generate

```bash
# Laptop
python examples/vllm_generate_embed/generate.py --engine transformers --uri ./llm_db

# GPU cluster
python examples/vllm_generate_embed/generate.py --engine vllm --uri ./llm_db \
  --model Qwen/Qwen2.5-1.5B-Instruct --concurrency 4
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--engine` | `vllm` | Generation backend |
| `--model` | per engine | Any chat model the backend supports |
| `--prompts` | built-in samples | File of prompts, one per line |
| `--uri` | `./llm_db` | LanceDB directory |
| `--table` | `answers` | Table name |
| `--max-tokens` | `128` | Generation length |
| `--concurrency` | `1` | Replicas of each model stage |
| `--batch-size` | `16` | Rows per batch |
| `--device` | `cpu` | Torch device for the local paths (`mps`, `cuda`) |

With no `--prompts`, a dozen built-in questions are used so the example runs out
of the box.

## 2. Search

```bash
streamlit run examples/vllm_generate_embed/answers_app.py --server.fileWatcherType none -- --uri ./llm_db
```

The `--` separates Streamlit's arguments from the script's.

`--server.fileWatcherType none` matters: Streamlit's default watcher walks
`sys.modules` and inspects every module's path, and because transformers resolves
attributes lazily, that inspection *imports* every image-processing module it
ships — most of which need `torchvision` — flooding the console with
`ModuleNotFoundError`. Turning the watcher off avoids it, at the cost of
auto-reload when you edit the app.

The app embeds your query with the same model the answers were embedded with, so
"how do I make search go faster" can surface an answer about indexing that never
uses those words. Click **Inspect** on any result for the full row, and
**Find similar answers** to search from that row's own embedding.

## Files

| File | Purpose |
| --- | --- |
| `generate.py` | The Ray pipeline: prompts → generate → embed → write |
| `answers_app.py` | Streamlit search UI |
| `text_embedding.py` | Shared embedding, used by both |
| `requirements.txt` | Extra dependencies |

`text_embedding.py` is shared deliberately: answers embedded at write time and
queries embedded at search time must land in the same space with the same
pooling and normalisation, so that lives in one place rather than in two files
that can drift.

## Notes

**Answers are embedded, not prompts.** The vector describes what the model
actually said. Searching prompts instead would just be searching your own
questions back.

**Embeddings are mean-pooled and L2-normalised.** Pooling is masked so padding
does not drag the average around, and normalising at write time makes cosine
similarity a plain dot product.

**No index is built.** A dozen answers is far below where an ANN index helps —
see the image-search example's README for why a small collection is better off
scanned exhaustively. On a large corpus, add `table.create_index(...)` after the
write.
