#!/usr/bin/env python3
"""HDR splatting, bloom and tone mapping for luminous attractor renders.

The original editions drew one-pixel polylines onto a black canvas. That reads
as neon-coloured line art, not as light. Here every sample is deposited into a
float accumulation buffer with bilinear weights, so overlapping trajectories
genuinely add up; dense regions blow past 1.0 and tone map towards white while
sparse regions stay saturated. A multi-scale bloom pass then bleeds the bright
cores outward, which is what survives Instagram's compression.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MONO_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
MONO_BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")


def build_palette(stops: list[tuple[int, int, int]], samples: int = 1024) -> np.ndarray:
    """Linear ramp through the colour stops, normalised to 0..1 floats."""
    positions = np.linspace(0, len(stops) - 1, samples)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (positions - lower)[:, None]
    ramp = (1 - fraction) * np.asarray(stops, dtype=np.float64)[lower]
    ramp += fraction * np.asarray(stops, dtype=np.float64)[upper]
    return (ramp / 255.0).astype(np.float32)


def sample_palette(palette: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Look up palette colours for values already normalised to 0..1."""
    indices = np.clip(values, 0.0, 1.0) * (len(palette) - 1)
    return palette[indices.astype(np.int32)]


def splat(
    width: int,
    height: int,
    screen: np.ndarray,
    colours: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Additively deposit coloured samples with bilinear sub-pixel weights.

    Returns the weighted colour sum and the density (weight sum) separately, so
    the tone mapper can divide one by the other and recover the average hue of
    a pixel independently of how many samples landed on it.

    Sub-pixel placement is what keeps a slowly rotating cloud from crawling and
    shimmering: without it every sample snaps to an integer pixel and the whole
    structure visibly jitters as the camera turns.
    """
    x, y = screen[:, 0], screen[:, 1]
    x_floor = np.floor(x)
    y_floor = np.floor(y)
    x_fraction = (x - x_floor).astype(np.float32)
    y_fraction = (y - y_floor).astype(np.float32)
    x_base = x_floor.astype(np.int64)
    y_base = y_floor.astype(np.int64)

    colour_sum = np.zeros((height * width, 3), dtype=np.float32)
    density = np.zeros(height * width, dtype=np.float32)
    for offset_x in (0, 1):
        for offset_y in (0, 1):
            pixel_x = x_base + offset_x
            pixel_y = y_base + offset_y
            share = (x_fraction if offset_x else 1.0 - x_fraction) * (
                y_fraction if offset_y else 1.0 - y_fraction
            )
            share = share * weights
            inside = (
                (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
            )
            if not inside.any():
                continue
            flat_index = pixel_y[inside] * width + pixel_x[inside]
            share_inside = share[inside]
            colour_inside = colours[inside]
            density += np.bincount(
                flat_index, weights=share_inside, minlength=width * height
            ).astype(np.float32)
            for channel in range(3):
                colour_sum[:, channel] += np.bincount(
                    flat_index,
                    weights=share_inside * colour_inside[:, channel],
                    minlength=width * height,
                ).astype(np.float32)
    return colour_sum.reshape(height, width, 3), density.reshape(height, width)


def flame_map(
    colour_sum: np.ndarray,
    density: np.ndarray,
    reference: float,
    boost: float = 1.0,
) -> np.ndarray:
    """Log-density tone mapping in the spirit of fractal flames.

    An attractor's density spans several orders of magnitude -- Lorenz spends
    almost all of its time in the two wing cores. Mapping it linearly blows
    those cores to featureless white and leaves the surrounding filigree in the
    dark. Taking the log compresses that range so the faint structure reads
    while the cores stay bright, and dividing the colour sum by the density
    keeps the hue saturated instead of washing everything towards white.
    """
    safe_density = np.maximum(density, 1e-12)
    average_colour = colour_sum / safe_density[:, :, None]
    brightness = np.log1p(density * (1.0 / max(reference, 1e-9)) * 12.0) / math.log(13.0)
    return average_colour * np.clip(brightness * boost, 0.0, None)[:, :, None]


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via cumulative sums; three passes approximate a gaussian."""
    if radius < 1:
        return image
    blurred = image
    for axis in (0, 1):
        padded = np.pad(
            blurred,
            [(radius + 1, radius) if index == axis else (0, 0) for index in range(3)],
            mode="edge",
        )
        cumulative = np.cumsum(padded, axis=axis, dtype=np.float32)
        leading = np.take(cumulative, np.arange(2 * radius + 1, cumulative.shape[axis]), axis=axis)
        trailing = np.take(cumulative, np.arange(0, cumulative.shape[axis] - 2 * radius - 1), axis=axis)
        blurred = (leading - trailing) / (2 * radius + 1)
    return blurred


def _downsample(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    height -= height % 2
    width -= width % 2
    trimmed = image[:height, :width]
    return trimmed.reshape(height // 2, 2, width // 2, 2, 3).mean(axis=(1, 3))


def _upsample(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Double the image until it covers the target shape, then crop to it."""
    grown = image
    while grown.shape[0] < shape[0] or grown.shape[1] < shape[1]:
        grown = np.repeat(np.repeat(grown, 2, axis=0), 2, axis=1)
    return grown[: shape[0], : shape[1]]


def bloom(
    image: np.ndarray,
    threshold: float = 0.32,
    levels: int = 6,
    strength: float = 0.55,
) -> np.ndarray:
    """Multi-scale glow built on a down-then-up pyramid.

    Only the part of the image above `threshold` is allowed to glow, so the
    halo comes from genuinely bright structure rather than lifting the whole
    frame into grey haze. Walking back up the pyramid one doubling at a time
    and blurring after each step is what keeps the halo smooth -- upsampling
    a tiny level straight to full resolution leaves visible square blocking.
    """
    height, width = image.shape[:2]
    # Every level must halve exactly. 1080 reaches an odd number after three
    # halvings, and dropping the odd row each time walks the coarse levels out
    # of alignment with the fine ones -- which shows up as a soft halo sitting
    # visibly offset from the structure that cast it.
    block = 1 << levels
    padded_height = math.ceil(height / block) * block
    padded_width = math.ceil(width / block) * block
    bright = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    bright[:height, :width] = np.maximum(image - threshold, 0.0)

    pyramid = [bright]
    for _ in range(levels):
        pyramid.append(_box_blur(_downsample(pyramid[-1]), 2))

    glow = np.zeros_like(pyramid[-1])
    for level in range(len(pyramid) - 1, 0, -1):
        glow = _box_blur(_upsample(glow + pyramid[level], pyramid[level - 1].shape[:2]), 2)
        glow *= 0.62
    return image + glow[:height, :width] * strength


def tone_map(image: np.ndarray, exposure: float = 1.0, gamma: float = 2.2) -> np.ndarray:
    """Exponential roll-off to 0..1, then gamma encode to display space."""
    mapped = 1.0 - np.exp(-np.maximum(image, 0.0) * exposure)
    return np.power(mapped, 1.0 / gamma, dtype=np.float32)


def to_bytes(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def make_caption(
    width: int,
    height: int,
    title: str,
    equation: tuple[str, ...],
    equation_size: int = 27,
    title_size: int = 30,
    margin: int = 64,
    top_margin: int | None = None,
    bottom_margin: int | None = None,
    scrim: float = 0.0,
    equation_face: Path | None = None,
) -> Image.Image:
    """Transparent caption layer: spaced title top-left, data block bottom-left.

    The block sits on the left where the eye lands first and carries a soft
    shadow so it survives against the bright parts of the render.

    Top and bottom insets are separate from the left one: the Reel player puts
    its header over the top of the frame and its controls over the bottom, and
    the two need different amounts of clearance.

    `equation_face` overrides the face used for the data block alone. A caption
    carrying Greek has to fall back to DejaVu for those glyphs, but the title
    and everything else in the frame should stay in Plex; without the override
    one Greek letter drags the whole layer onto the fallback face.
    """
    if not MONO_FONT.exists() or not MONO_BOLD_FONT.exists():
        raise RuntimeError(f"DejaVu Sans Mono is required: {MONO_FONT}")
    equation_font = ImageFont.truetype(str(equation_face or MONO_FONT), equation_size)
    title_font = ImageFont.truetype(str(MONO_BOLD_FONT), title_size)

    top = margin if top_margin is None else top_margin
    bottom = margin if bottom_margin is None else bottom_margin

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if scrim > 0.0:
        # A soft darkening under the text, strongest at the very edge and gone
        # well before the middle of the frame. Pieces that fill the frame leave
        # no black for the caption to sit on, and a stroke alone does not hold
        # up against a bright pattern; this stays under the reading without
        # ever reading as a box.
        ramp = np.zeros((height, 1), dtype=np.float32)
        top_band = max(1, int(height * 0.20))
        bottom_band = max(1, int(height * 0.26))
        ramp[:top_band, 0] = np.linspace(1.0, 0.0, top_band) ** 1.25
        ramp[height - bottom_band :, 0] = np.linspace(0.0, 1.0, bottom_band) ** 1.25
        veil = np.zeros((height, width, 4), dtype=np.uint8)
        veil[:, :, 3] = (ramp * (255.0 * scrim)).astype(np.uint8)
        overlay.alpha_composite(Image.fromarray(veil, "RGBA"))
    draw = ImageDraw.Draw(overlay)

    spaced_title = " ".join(title.upper())
    draw.text(
        (margin, top),
        spaced_title,
        font=title_font,
        fill=(255, 255, 255, 236),
        stroke_width=4,
        stroke_fill=(0, 0, 0, 150),
    )

    text = "\n".join(equation)
    spacing = max(4, equation_size // 3)
    box = draw.multiline_textbbox((0, 0), text, font=equation_font, spacing=spacing)
    # Measured from the drawing origin, not the ink box: multiline_text anchors
    # the origin, so subtracting the ink height alone lets the descenders and
    # the stroke run past the margin by however far the ink sits below zero.
    draw.multiline_text(
        (margin, height - bottom - box[3] - 4),
        text,
        font=equation_font,
        fill=(228, 236, 248, 224),
        spacing=spacing,
        align="left",
        stroke_width=4,
        stroke_fill=(0, 0, 0, 150),
    )
    return overlay


def compose(frame: np.ndarray, caption: Image.Image) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGBA")
    image.alpha_composite(caption)
    return np.asarray(image.convert("RGB"))
