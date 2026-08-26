# SPDX-License-Identifier: Apache-2.0
"""Generate a directory of synthetic JPGs so the example runs without photos.

These are coloured shapes on plain backgrounds, not photographs, so CLIP will
not produce anything like the semantic matches you would get from a real photo
library. They exist so the pipeline can be exercised end to end -- scan, embed,
write, index, search -- before pointing it at real data.

Run with::

    python examples/clip_image_search/make_sample_images.py --out ./sample_images
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

#: (name, RGB) pairs used for both backgrounds and shapes.
COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (200, 30, 30)),
    ("green", (30, 160, 60)),
    ("blue", (40, 70, 200)),
    ("yellow", (230, 200, 40)),
    ("purple", (130, 50, 180)),
    ("orange", (230, 130, 30)),
)

SHAPES: tuple[str, ...] = ("circle", "square", "triangle")


def draw_shape(
    draw: ImageDraw.ImageDraw, shape: str, color: tuple[int, int, int], size: int
) -> None:
    """Draw one centred shape covering roughly half the canvas."""
    pad = size // 4
    box = (pad, pad, size - pad, size - pad)
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    else:
        draw.polygon(
            [(size // 2, pad), (size - pad, size - pad), (pad, size - pad)], fill=color
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./sample_images")
    parser.add_argument("--size", type=int, default=224, help="Pixels per side")
    parser.add_argument(
        "--repeats",
        type=int,
        default=6,
        help="Copies of each colour/shape combination, varying the background",
    )
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for shape in SHAPES:
        for color_name, color in COLORS:
            for i in range(args.repeats):
                # Vary the background so identical shapes are not identical files.
                shade = 240 - (i * 25)
                image = Image.new("RGB", (args.size, args.size), (shade, shade, shade))
                draw_shape(ImageDraw.Draw(image), shape, color, args.size)
                image.save(out / f"{color_name}_{shape}_{i}.jpg", quality=90)
                count += 1

    print(f"Wrote {count} JPGs to {out}")
    print(f"\nNext:\n  python examples/clip_image_search/ingest.py --images {out}")


if __name__ == "__main__":
    main()
