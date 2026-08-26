# SPDX-License-Identifier: Apache-2.0
"""Shared CLIP embedding logic for the image-search example.

Both the ingestion job and the Streamlit app need to produce vectors in the
same space -- images at write time, text queries at search time -- so the model
name, dimensionality and normalisation live here rather than being duplicated
and drifting apart.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

#: A small, fast, widely available CLIP checkpoint. Image and text encoders
#: share an embedding space, which is what makes text-to-image search work.
DEFAULT_MODEL = "openai/clip-vit-base-patch32"

#: Embedding width for :data:`DEFAULT_MODEL`. Declared up front because the
#: LanceDB column is a fixed-size list and the schema is built before any
#: vector exists.
EMBEDDING_DIM = 512


def load_clip(model_name: str = DEFAULT_MODEL, device: str = "cpu") -> tuple[Any, Any]:
    """Load a CLIP model and its processor, ready for inference."""
    from transformers import CLIPModel, CLIPProcessor

    # Load the concrete CLIP classes rather than AutoModel/AutoProcessor. The
    # Auto* classes resolve through transformers' model registry, which can
    # route via AutoImageProcessor and import unrelated model modules -- several
    # of which require torchvision and raise ModuleNotFoundError when it is not
    # installed, even though CLIP itself never needs it. Naming the classes
    # skips that discovery entirely, and is faster besides.
    # Annotated Any deliberately: transformers' own annotations vary between
    # versions (and are absent entirely when it is not installed, as in lint
    # CI), so type-checking these calls would pass in one environment and fail
    # in the other. Everything below uses the objects dynamically anyway.
    processor: Any = CLIPProcessor.from_pretrained(model_name)
    model: Any = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor


def _as_tensor(features: Any) -> Any:
    """Pull the embedding tensor out of whatever the model returned.

    transformers 4.x returned a bare tensor from ``get_image_features`` /
    ``get_text_features``; 5.x wraps it in a ``BaseModelOutputWithPooling``
    whose ``pooler_output`` holds the projected vector. Handling both keeps the
    example working across the version range in requirements.txt.
    """
    import torch

    if isinstance(features, torch.Tensor):
        return features
    pooled = getattr(features, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(
        f"Unexpected CLIP output {type(features).__name__}; expected a tensor "
        "or an output object exposing 'pooler_output'."
    )


def to_unit_vectors(features: Any) -> npt.NDArray[np.float32]:
    """Normalise a batch of feature tensors and hand back a NumPy array.

    CLIP similarity is cosine similarity. Normalising once at write time means
    the index compares plain distances and every stored vector is directly
    comparable.

    The normalisation runs in Torch, on whatever device the features are
    already on, rather than after copying to the host: at a 2048-row batch that
    is ~3.5x faster on CPU and ~2.4x on MPS. ``F.normalize`` also clamps by
    ``eps`` internally, so an all-zero row yields zeros instead of NaNs without
    a separate guard.

    Note this is not where a run's time goes -- the forward pass dominates by
    orders of magnitude -- but it is free to do correctly.
    """
    import torch.nn.functional as F

    unit = F.normalize(_as_tensor(features), p=2, dim=-1)
    # CLIP already emits float32, so copy=False makes astype a no-op rather
    # than an extra full-array copy.
    return unit.cpu().numpy().astype(np.float32, copy=False)  # type: ignore[no-any-return]


def embed_images(
    model: Any, processor: Any, images: list[Any], device: str = "cpu"
) -> npt.NDArray[np.float32]:
    """Embed PIL images into the shared CLIP space."""
    import torch

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # inference_mode skips autograd bookkeeping entirely; no_grad only disables
    # gradient recording.
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
    return to_unit_vectors(features)


def embed_text(
    model: Any, processor: Any, queries: list[str], device: str = "cpu"
) -> npt.NDArray[np.float32]:
    """Embed text into the same space the images were embedded into."""
    import torch

    inputs = processor(text=queries, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
    return to_unit_vectors(features)


def decode_image(payload: bytes) -> Any:
    """Decode JPEG bytes into an RGB PIL image."""
    from PIL import Image

    image = Image.open(io.BytesIO(payload))
    # CLIP's processor expects 3 channels; greyscale and CMYK JPEGs exist.
    return image.convert("RGB")
