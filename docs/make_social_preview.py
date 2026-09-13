#!/usr/bin/env python3
"""Build `social-preview.png` -- the card GitHub shows when this repo's link is
pasted somewhere else.

It never appears on the repository page itself. It is the Open Graph image: X,
LinkedIn, Slack, Discord, Messenger and iMessage all turn the link into a card
and use this as the picture. GitHub has no API for setting it, so it has to be
uploaded by hand in Settings -> General -> Social preview.

Two things are done to the cover stills before they are laid on the card, and
both were measured rather than guessed:

* **The bloom leaves a tinted pedestal at the frame edge.** Sampled over the
  outer eight pixels, one cover averages (2.9, 3.4, 1.2) and another
  (3.3, 1.8, 2.7) -- greenish and warm respectively, against the card's true
  black. Pasted straight down, each still reads as a faintly glowing rectangle.
  A soft ramp to zero below `FLOOR` removes it.
* **The tile edge is feathered** over `FEATHER` pixels, so the boundary between
  still and card dissolves instead of ruling a line.

The halo around the subject itself is left alone. That is the render's own
multi-scale bloom, not an artefact -- brightness is how much stuff is there,
and removing it would mean changing the piece.

    python3 make_social_preview.py

Needs Pillow and numpy, and the cover stills the pieces write into `../out/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONTS = HERE.parent / "src" / "fonts"
OUT = HERE.parent / "out"

WIDTH, HEIGHT = 1280, 640
TILE = (300, 533)
FEATHER = 38
FLOOR = 14

# Three of the six on the front page, chosen for contrast: a spiral, a solid
# disc and a scattered field, in violet, gold and every colour at once. The card
# is used in the README as well as by GitHub, so it shows pieces the page
# actually contains rather than advertising work that is not there.
PICKS = [
    "phyllotaxis_primordia_apex_1080x1920_8s_30fps_hook_plex.cover.png",
    "stripe_pigment-cells_skin_1080x1920_8s_30fps_bloom_hook_plex.cover.png",
    "tear_placozoan-spread_prism_1080x1920_10s_30fps_bloom_hook_plex.cover.png",
]


def clean(path: Path, size: tuple[int, int]) -> Image.Image:
    """Floor the bloom pedestal to black, then feather the tile's own edges."""
    array = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    luminance = array.max(2)
    # Smooth rather than a hard cut: a step at FLOOR would band across the
    # gradient the bloom leaves behind.
    array *= np.clip(luminance / FLOOR, 0.0, 1.0)[..., None] ** 1.6
    tile = Image.fromarray(array.clip(0, 255).astype(np.uint8)).resize(size, Image.LANCZOS)

    width, height = size
    across = np.clip(np.minimum(np.arange(width), width - 1 - np.arange(width)) / FEATHER, 0, 1)
    down = np.clip(np.minimum(np.arange(height), height - 1 - np.arange(height)) / FEATHER, 0, 1)
    mask = np.minimum(across[None, :], down[:, None])[..., None]
    return Image.fromarray((np.asarray(tile).astype(np.float32) * mask).clip(0, 255).astype(np.uint8))


def main() -> None:
    card = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    tile_w, tile_h = TILE
    left = WIDTH - 3 * tile_w - 30
    for index, name in enumerate(PICKS):
        source = OUT / name
        if not source.exists():
            raise SystemExit(f"Missing cover: {source}\nRender the piece first; see ../src/README.md.")
        card.paste(clean(source, TILE), (left + index * (tile_w + 10), (HEIGHT - tile_h) // 2))

    draw = ImageDraw.Draw(card)
    bold = ImageFont.truetype(str(FONTS / "IBMPlexMono-Bold.ttf"), 52)
    regular = ImageFont.truetype(str(FONTS / "IBMPlexMono-Regular.ttf"), 23)
    small = ImageFont.truetype(str(FONTS / "IBMPlexMono-Regular.ttf"), 19)

    draw.text((56, 214), "ON GROWTH", font=bold, fill=(236, 236, 236))
    draw.text((56, 276), "AND FORM", font=bold, fill=(236, 236, 236))
    draw.line([(56, 356), (300, 356)], fill=(90, 90, 90), width=2)
    draw.text((56, 380), "Biology is the original", font=regular, fill=(168, 168, 168))
    draw.text((56, 410), "algorithm", font=regular, fill=(168, 168, 168))
    draw.text((56, 452), "@ekspertodniczego", font=small, fill=(228, 64, 95))

    destination = HERE / "social-preview.png"
    card.save(destination)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
