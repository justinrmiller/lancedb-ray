# SPDX-License-Identifier: Apache-2.0
"""Shared text embedding for the generate-and-embed example.

Both the pipeline (embedding generated answers) and the Streamlit app
(embedding a search query) need vectors in the same space, so the model name,
dimensionality and pooling live here rather than being duplicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

#: A small, fast sentence embedding model. 384 dimensions, CPU-friendly, and
#: good enough that semantic search over a few thousand answers is meaningful.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Embedding width for :data:`DEFAULT_EMBEDDING_MODEL`. Declared up front
#: because the LanceDB column is a fixed-size list and the table is created
#: before any text has been embedded.
EMBEDDING_DIM = 384


def load_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"
) -> tuple[Any, Any]:
    """Load a sentence embedding model and its tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    # Annotated Any: transformers' annotations vary across versions and are
    # absent entirely in the lint environment, so type-checking these calls
    # would pass in one and fail in the other.
    tokenizer: Any = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    model: Any = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def embed_text(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: str = "cpu",
    max_length: int = 512,
) -> npt.NDArray[np.float32]:
    """Embed a batch of strings into unit-length vectors.

    Sentence embeddings from this family are mean-pooled over tokens with the
    attention mask applied, so padding does not drag the average around, and
    then L2-normalised so cosine similarity is a plain dot product.
    """
    import torch
    import torch.nn.functional as F

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        output = model(**encoded)

    hidden = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    # Mean-pool over real tokens only.
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    unit = F.normalize(pooled, p=2, dim=-1)
    return unit.cpu().numpy().astype(np.float32, copy=False)  # type: ignore[no-any-return]
