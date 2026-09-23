# Examples

Each example lives in its own directory with a README explaining what it
demonstrates, how to run it, and what to look for in the output.

| Example | What it shows |
| --- | --- |
| [`quickstart/`](quickstart/) | The core guarantees: a write fans out across Ray tasks yet lands as a single atomic commit, and reads come back fragment-parallel. No external dependencies. |
| [`clip_image_search/`](clip_image_search/) | A realistic pipeline — scan a directory of JPGs, embed them with CLIP across a Ray cluster, write to LanceDB, build a vector index, then search them in plain English from a Streamlit app. |
| [`vllm_generate_embed/`](vllm_generate_embed/) | An LLM pipeline — answer prompts with vLLM (or a small local model), embed each answer, write them to LanceDB, then search what the model said by meaning from a Streamlit app. |
| [`mcap_ingest/`](mcap_ingest/) | Index robotics logs — read a directory of MCAP recordings in parallel (one task per file, streamed rather than loaded), write them to LanceDB as one atomic commit, then query the messages by topic and time window. |
| [`object_storage/`](object_storage/) | Verify writes to S3-compatible object storage — a Floci emulator in Docker Compose plus a large locally generated dataset, asserting the round trip and the single atomic commit. |

Start with `quickstart/` if you want to understand what the library guarantees.
Start with `clip_image_search/` or `vllm_generate_embed/` if you want to see
what it is actually *for*. Start with `mcap_ingest/` if your source is a pile
of files Ray cannot split.

## Running

Every example assumes `lancedb-ray` is installed:

```bash
make build
```

Examples with extra dependencies list them in their own `requirements.txt`.
