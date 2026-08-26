# SPDX-License-Identifier: Apache-2.0
"""Generate text with an LLM, embed it, and store the result in LanceDB.

The pipeline is four Ray Data stages:

1. Prompts become a Ray Dataset.
2. An LLM answers each prompt. With ``--engine vllm`` this is Ray Data's
   ``build_processor`` driving a vLLM engine across the cluster -- see
   https://docs.ray.io/en/latest/data/working-with-llms.html -- which needs a
   GPU. ``--engine transformers`` runs a small model locally instead, so the
   rest of the pipeline can be exercised on a laptop.
3. Each answer is embedded by a Ray actor holding a sentence-embedding model,
   loaded once per worker rather than once per batch.
4. ``write_lancedb`` writes prompt, answer and vector as one atomic commit.

Run with::

    # GPU cluster
    python examples/vllm_generate_embed/generate.py --engine vllm --uri ./llm_db

    # Laptop
    python examples/vllm_generate_embed/generate.py --engine transformers --uri ./llm_db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import ray
from lancedb_ray import write_lancedb

#: Shipped to Ray workers as the job's working_dir so the embedder actor can
#: import ``text_embedding`` wherever it is constructed.
EXAMPLE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXAMPLE_DIR))

from text_embedding import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    embed_text,
    load_embedder,
)

#: Default model for ``--engine vllm``. Any chat model vLLM supports works.
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

#: Default model for ``--engine transformers``. Deliberately tiny so the
#: example runs on CPU in a reasonable time.
DEFAULT_LOCAL_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

#: Used when no --prompts file is given, so the example runs out of the box.
SAMPLE_PROMPTS = (
    "What is a vector database, in two sentences?",
    "Explain what an embedding is to a backend engineer.",
    "Why would you use approximate nearest neighbour search?",
    "What does IVF_PQ stand for and what does it trade away?",
    "When is a brute-force vector scan faster than an index?",
    "Describe cosine similarity without using formulas.",
    "What problem does Apache Arrow solve?",
    "Why is columnar storage good for analytics?",
    "What is the difference between a data lake and a warehouse?",
    "Explain distributed data parallelism in one paragraph.",
    "What is backpressure in a streaming system?",
    "Why do batch sizes matter for GPU inference?",
)


def build_schema() -> pa.Schema:
    """Schema for the generated-text table.

    Declared explicitly: the vector column must be a fixed-size list for
    LanceDB to index it, and the table is created before anything is embedded.
    """
    return pa.schema(
        [
            pa.field("prompt", pa.string()),
            pa.field("response", pa.string()),
            pa.field("model", pa.string()),
            pa.field("num_chars", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ]
    )


def read_prompts(path: str | None) -> list[str]:
    """Load prompts from a file (one per line) or fall back to the samples."""
    if path is None:
        return list(SAMPLE_PROMPTS)
    lines = Path(path).expanduser().read_text().splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def generate_with_vllm(
    ds: ray.data.Dataset,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    batch_size: int,
) -> ray.data.Dataset:
    """Answer every prompt with vLLM, via Ray Data's LLM processor.

    Ray places one vLLM engine per replica and streams batches through them,
    so scaling up is a matter of raising ``concurrency`` on a bigger cluster.
    Requires GPUs; there is no CPU fallback inside this path by design, since
    a silent fallback would hide why throughput collapsed.
    """
    from ray.data.llm import build_processor, vLLMEngineProcessorConfig

    config = vLLMEngineProcessorConfig(
        model_source=model,
        concurrency=concurrency,
        batch_size=batch_size,
        engine_kwargs={"max_model_len": 2048},
    )

    processor = build_processor(
        config,
        preprocess=lambda row: {
            "messages": [{"role": "user", "content": row["prompt"]}],
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            # Carry the original question under a key the pipeline will not
            # touch. The chat-template stage overwrites "prompt" with the
            # fully templated text -- system turn, role markers and all -- so
            # reading it back in postprocess would store that instead of the
            # question the user actually asked.
            "user_prompt": row["prompt"],
        },
        postprocess=lambda row: {
            "prompt": row["user_prompt"],
            "response": row["generated_text"],
            "model": model,
        },
    )
    return processor(ds)


class LocalGenerator:
    """Answer prompts with a small local model, one load per worker.

    The point of the class (rather than a function) is that Ray keeps the
    actor alive between batches, so the model is loaded once per worker.
    """

    def __init__(self, model_name: str, max_tokens: int, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device
        # Locals annotated Any: transformers' annotations vary across versions
        # and are absent entirely in the lint environment, so type-checking
        # these calls would pass in one and fail in the other.
        tokenizer: Any = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        import torch

        prompts = [str(p) for p in batch["prompt"]]
        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        encoded = self.tokenizer(
            chats, return_tensors="pt", padding=True, padding_side="left"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # generate() returns the prompt followed by the completion; keep only
        # the newly generated tail.
        prompt_len = encoded["input_ids"].shape[1]
        responses = [
            self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            for row in generated
        ]
        return {
            "prompt": np.array(prompts, dtype=object),
            "response": np.array(responses, dtype=object),
            "model": np.array([self.model_name] * len(prompts), dtype=object),
        }


class TextEmbedder:
    """Embed generated answers, holding the model across batches."""

    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        self.model, self.tokenizer = load_embedder(model_name, device)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        responses = [str(r) for r in batch["response"]]
        vectors = embed_text(self.model, self.tokenizer, responses, self.device)
        return {
            "prompt": np.asarray(batch["prompt"], dtype=object),
            "response": np.array(responses, dtype=object),
            "model": np.asarray(batch["model"], dtype=object),
            "num_chars": np.array([len(r) for r in responses], dtype=np.int64),
            "vector": vectors,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("vllm", "transformers"),
        default="vllm",
        help="vllm needs GPUs; transformers runs a small model locally",
    )
    parser.add_argument("--model", default=None, help="Generation model")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--prompts", default=None, help="File of prompts, one per line")
    parser.add_argument("--uri", default="./llm_db", help="LanceDB directory")
    parser.add_argument("--table", default="answers")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="cpu", help="Torch device for the local paths"
    )
    args = parser.parse_args()

    model = args.model or (
        DEFAULT_VLLM_MODEL if args.engine == "vllm" else DEFAULT_LOCAL_MODEL
    )
    prompts = read_prompts(args.prompts)
    uri = str(Path(args.uri).expanduser().resolve())

    print(f"{len(prompts)} prompts, generating with {model} via {args.engine}")

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"working_dir": str(EXAMPLE_DIR)},
    )
    started = time.perf_counter()
    try:
        ds = ray.data.from_items([{"prompt": p} for p in prompts])

        if args.engine == "vllm":
            answered = generate_with_vllm(
                ds,
                model,
                args.max_tokens,
                args.temperature,
                args.concurrency,
                args.batch_size,
            )
        else:
            answered = ds.map_batches(
                LocalGenerator,
                batch_size=args.batch_size,
                compute=ray.data.ActorPoolStrategy(size=args.concurrency),
                fn_constructor_kwargs={
                    "model_name": model,
                    "max_tokens": args.max_tokens,
                    "device": args.device,
                },
            )

        embedded = answered.map_batches(
            TextEmbedder,
            batch_size=args.batch_size,
            compute=ray.data.ActorPoolStrategy(size=args.concurrency),
            fn_constructor_kwargs={
                "model_name": args.embedding_model,
                "device": args.device,
            },
        )

        write_lancedb(
            embedded, args.table, uri=uri, mode="overwrite", schema=build_schema()
        )

        table = lancedb.connect(uri).open_table(args.table)
        elapsed = time.perf_counter() - started
        print(
            f"  generated, embedded and wrote {table.count_rows()} rows in {elapsed:.1f}s"
        )
        print(
            f"\nExplore it with:\n"
            f"  streamlit run examples/vllm_generate_embed/answers_app.py "
            f"--server.fileWatcherType none -- --uri {uri} --table {args.table}"
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
