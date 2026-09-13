"""Phyllotaxis -- a shoot apex placing organs where the ones already there object least.

One rule, applied once per plastochrone: look around the rim of the growing tip
and start the next primordium at whatever angle is furthest, in the inhibitory
sense, from the primordia already placed; then let the tissue underneath grow,
which carries every existing primordium outwards and clears the rim for the
next one. Douady & Couder, 1992.

Nothing in that rule mentions an angle, a spiral or a number. What comes out is
the golden angle and two families of counter-rotating spirals whose counts are
consecutive Fibonacci numbers.

This is the code that rendered the clip in ../reels/phyllotaxis/. It is the
model and the frame pipeline as shipped, with the edition registry and the
other pieces stripped out.

    python3 phyllotaxis.py --preview        # one still
    python3 phyllotaxis.py                  # the 8 s clip

Needs numpy, Pillow, and an ffmpeg built with libx264.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import glow

# Violet through magenta, with the youngest few percent of the ramp swapping to
# cyan -- colour is age and the newest organs are the smallest ring, right at
# the centre, so the core reads as a complementary accent against the arms.
APEX = [(8, 0, 26), (72, 0, 140), (190, 0, 170), (255, 60, 190), (80, 230, 255), (215, 255, 255)]

SPEC = {
    "title": "Phyllotaxis",
    "slug": "phyllotaxis_primordia_apex",
    "palette": APEX,
    "exposure": 1.16,
    "boost": 1.24,
    "steps_per_frame": 5,
    "cell_radius": 9.0,
    "cell_samples": 48,
    "caption": (
        "shoot apical meristem  \u00b7  Douady & Couder 1992",
        "grow \u00b7 inhibit \u00b7 place the next one",
        "spiral counts 8, 13, 21 out to 21, 34, 55",
        "divergence 137.4\u00b0  \u00b7  no angle is in the rule",
    ),
    "hook": ("The plant is not counting. You are.",),
}

class Phyllotaxis:
    """A meristem placing organs where the ones already there object least.

    One rule, applied once per plastochrone: look around the rim of the
    growing tip and start the next primordium at whatever angle is furthest,
    in the inhibitory sense, from the primordia already placed. Then let the
    tissue underneath grow, which carries every existing primordium outwards
    and clears the rim for the next one. Douady and Couder, 1992.

    Nothing in that rule mentions an angle, a spiral or a number. What comes
    out is the golden angle and two families of counter-rotating spirals whose
    counts are consecutive Fibonacci numbers -- and, because the rim gets
    relatively flatter as the head widens, the counts climb the sequence as
    you read outwards, which is why a real sunflower has few spirals near its
    centre and many at its edge.

    Growth is r proportional to the square root of age, which is the only
    choice that keeps the areal density constant: organs are added at a steady
    rate, so the area they occupy has to grow at a steady rate too. It is also
    what makes the head fill out rather than thin as it widens.

    The same authors got this pattern out of ferrofluid drops repelling each
    other in a dish of oil, with no biology in it at all, which is the useful
    thing to remember about it.
    """

    def __init__(
        self,
        height: int,
        width: int,
        radius: float = 500.0,
        centre: tuple[float, float] | None = None,
        primordia: int = 1196,
        apex: float = 0.8,
        neighbours: int = 60,
        samples: int = 1440,
        falloff: float = 2.0,
        seed: int = 20260829,
    ) -> None:
        self.centre = centre if centre is not None else (width * 0.5, height * 0.448)
        # Area per organ, fixed, so that the oldest organ is at the head's edge
        # once the run has finished and the density is the same throughout.
        self.area = math.pi * radius * radius / primordia
        self.spacing = math.sqrt(self.area)
        # The apex has a size of its own and keeps it: organs are laid down on
        # a rim of fixed radius and it is the tissue underneath that expands.
        #
        # This is the parameter, and getting it wrong is visible. The rule has
        # ordered windows separated by disordered ones, and each ordered window
        # sits on a different angle: at 1.0 the pattern locks near five
        # thirteenths of a turn and throws the organs into thirteen separate
        # arms with gaps between them, at 1.2 onto a different fraction again,
        # and above about 1.4 the placement stops settling at all. At 0.8 the
        # angle it finds is 137.37 degrees, which is the golden angle to within
        # a seventh of a degree. **Measured, not assumed** -- nothing in the
        # rule refers to an angle, so the only way to know which window a
        # setting is in is to run it and take the median.
        self.apex = apex * self.spacing
        self.neighbours = neighbours
        self.falloff = falloff
        self.generator = np.random.default_rng(seed)
        self.grid = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
        self.angles = np.zeros(0, dtype=np.float64)
        self.count = 0

    def radii(self) -> np.ndarray:
        """Oldest first. Age in plastochrones is the only clock here.

        An organ of age a has a - 1 younger organs inside it, each holding the
        same area, so it has been carried out to wherever the annulus between
        the apex and itself has exactly that much room. The head therefore
        widens as the square root of its age -- the only law that adds area at
        the rate organs are added.
        """
        age = np.arange(self.count, 0, -1, dtype=np.float64)
        return np.sqrt(self.apex * self.apex + (age - 1.0) * self.area / math.pi)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            # The candidate ring is rotated by a fraction of its own spacing
            # each turn. Without it the answer can only ever be one of a fixed
            # set of angles, and the pattern locks to the sampling grid rather
            # than to the rule.
            offset = self.generator.random() * (self.grid[1] - self.grid[0])
            candidates = self.grid + offset
            if self.count == 0:
                self.angles = np.array([candidates[0]])
                self.count = 1
                continue

            # Only the youngest few matter: the inhibition falls off as a
            # cube, and everything older has been carried out of range by the
            # growth underneath it.
            near = min(self.neighbours, self.count)
            radii = self.radii()[-near:]
            angles = self.angles[-near:]
            x, y = radii * np.cos(angles), radii * np.sin(angles)
            offset_x = self.apex * np.cos(candidates)[:, None] - x[None, :]
            offset_y = self.apex * np.sin(candidates)[:, None] - y[None, :]
            square = np.maximum(offset_x * offset_x + offset_y * offset_y, 1e-9)
            inhibition = (square ** (-0.5 * self.falloff)).sum(axis=1)
            self.angles = np.append(self.angles, candidates[int(inhibition.argmin())])
            self.count += 1

    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions, and how recently each organ was laid down.

        Same quantity `folding` and `turing` are coloured by -- when it grew --
        which here runs the other way round: the youngest organ is the one in
        the middle, and the rim is the oldest thing in the picture.
        """
        radii = self.radii()
        points = np.column_stack((
            self.centre[0] + radii * np.cos(self.angles),
            self.centre[1] + radii * np.sin(self.angles),
        )).astype(np.float32)
        shade = (np.arange(self.count) / max(self.count - 1, 1)).astype(np.float32)
        return points, shade



# A cell is a few square microns of tissue, not a point, so it is splatted as
# a small blob. The offsets are drawn once and indexed by cell, so a cell keeps
# the same speckle from frame to frame -- redrawing them every frame makes the
# whole sheet boil.
DISC = np.column_stack((lambda a, r: (r * np.cos(a), r * np.sin(a)))(
    np.random.default_rng(7).uniform(0.0, 2.0 * math.pi, 8192),
    np.random.default_rng(11).random(8192) ** 0.62,
)).astype(np.float32)

def cell_samples(model, args, spec=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand one cell per row into a blob of samples carrying its phase.

    A somite cell is a few square microns and a floret is an organ, so the two
    want blobs an order of magnitude apart; the edition may say so.
    """
    spec = spec or {}
    points, shade = model.cells()
    radius = spec.get("cell_radius", args.cell_radius)
    count, per_cell = len(points), spec.get("cell_samples", args.cell_samples)
    index = (np.arange(count, dtype=np.int64)[:, None] * per_cell
             + np.arange(per_cell, dtype=np.int64)[None, :]) % len(DISC)
    samples = (points[:, None, :] + DISC[index] * radius).reshape(-1, 2)
    weights = np.full(count * per_cell, 1.0 / per_cell, dtype=np.float32)
    return samples, np.repeat(shade, per_cell), weights

def find_encoder() -> tuple[str, list[str]]:
    """Pick an ffmpeg that can actually encode H.264.

    Conda environments routinely ship one built without libx264; it advertises
    libopenh264 and then fails at runtime with a version mismatch.
    """
    candidates: list[str] = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / "ffmpeg"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            candidates.append(str(candidate))
    candidates.append("/usr/bin/ffmpeg")
    fallback: tuple[str, list[str]] | None = None
    for ffmpeg in dict.fromkeys(candidates):
        try:
            encoders = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        if " libx264 " in encoders:
            return ffmpeg, ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high"]
        if fallback is None and " libopenh264 " in encoders:
            fallback = (ffmpeg, ["-c:v", "libopenh264", "-threads", "1", "-b:v", "12M", "-maxrate", "16M"])
    if fallback is not None:
        return fallback
    raise RuntimeError("No ffmpeg with an H.264 encoder was found.")

def start_encoder(output: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    if width % 2 or height % 2:
        # yuv420p subsamples chroma by two; an odd dimension makes libx264 fail
        # with a message that says nothing about the actual cause.
        raise ValueError(f"H.264 needs even dimensions; got {width}x{height}.")
    ffmpeg, codec = find_encoder()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)

def build_overlay(width: int, height: int, spec: dict, args) -> Image.Image:
    """Title, hook and data block -- the layout the substrate set settled on.

    Plex throughout, spaced bold title 240 px down, data block 190 px up, hook
    centred at 34 px with its lowest ink 82 px clear of the block, soft scrim at
    both edges. Reused rather than re-derived: the numbers were measured off the
    `cleavage` cut pixel by pixel and there is no reason for a second opinion.
    """
    overlay = glow.make_caption(
        width,
        height,
        spec["title"],
        spec["caption"],
        equation_size=args.caption_size,
        margin=args.margin,
        top_margin=args.title_top,
        bottom_margin=args.caption_bottom,
        # A piece whose text already sits on black does not need the full
        # veil, and `somite` pays for it: the column runs to the top edge, so
        # a scrim strong enough to protect a title over texture also swallows
        # the oldest segments in the piece.
        scrim=spec.get("scrim", args.scrim),
    )
    lines = spec.get("hook") if args.hook else None
    if not lines:
        return overlay

    ink_top = glow.caption_ink_top(height, spec["caption"], args.caption_size, args.caption_bottom)
    font = ImageFont.truetype(str(glow.MONO_FONT), args.hook_size)
    draw = ImageDraw.Draw(overlay)
    text = "\n".join(lines)
    spacing = max(6, args.hook_size // 3)
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    draw.multiline_text(
        ((width - (box[2] - box[0])) // 2, ink_top - args.hook_gap - box[3]),
        text,
        font=font,
        fill=(255, 255, 255, 244),
        spacing=spacing,
        align="center",
        stroke_width=4,
        stroke_fill=(0, 0, 0, 165),
    )
    return overlay

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "out")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--duration", type=float, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--head-radius", type=float, default=500.0)
    # A cell is a few square microns of tissue, not a point, so it is splatted
    # as a small blob. SPEC overrides both for this piece.
    parser.add_argument("--cell-radius", type=float, default=3.1)
    parser.add_argument("--cell-samples", type=int, default=10)
    parser.add_argument("--preview", action="store_true", help="Save the cover still and stop.")
    parser.add_argument("--bloom-threshold", type=float, default=0.30)
    parser.add_argument("--bloom-strength", type=float, default=0.60)
    # Typography. Measured off the shipped layout pixel by pixel; see build_overlay.
    parser.add_argument("--margin", type=int, default=64)
    parser.add_argument("--title-top", type=int, default=240)
    parser.add_argument("--caption-bottom", type=int, default=190)
    parser.add_argument("--caption-size", type=int, default=27)
    parser.add_argument("--scrim", type=float, default=0.95)
    parser.add_argument("--no-hook", dest="hook", action="store_false")
    parser.add_argument("--hook-size", type=int, default=34)
    parser.add_argument("--hook-gap", type=int, default=82)
    # The percentile of the finished form's density that tone mapping is
    # calibrated against, so the clip brightens into its final state instead of
    # being levelled frame by frame.
    parser.add_argument("--cell-reference", type=float, default=92.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width, height = args.width, args.height
    frames = round(args.duration * args.fps)
    palette = glow.build_palette(SPEC["palette"])
    caption = build_overlay(width, height, SPEC, args)

    # The head is exactly full on the last step, which is what fixes the
    # spacing: the clip is the whole ontogeny of the head, not a window onto
    # part of it. `primordia` sets the area per organ; the settle below is one
    # step shorter, because frame zero is the cover and the run starts after it.
    organs = SPEC["steps_per_frame"] * (frames - 1) + 1
    model = Phyllotaxis(height, width, radius=args.head_radius, primordia=organs)

    # First pass: run the process to completion. The finished form is both the
    # cover frame and what the exposure is calibrated on, so that the clip
    # brightens into its final state instead of being levelled frame by frame.
    print("  settling", flush=True)
    model.step(SPEC["steps_per_frame"] * (frames - 1))
    samples, _, weights = cell_samples(model, args, SPEC)
    _, probe = glow.splat(width, height, samples, np.zeros((len(samples), 3), dtype=np.float32), weights)
    reference = float(np.percentile(probe[probe > 0], args.cell_reference))
    print(f"  {model.count:,} organs, outer radius {model.radii()[0]:.0f} px", flush=True)

    def draw(state) -> np.ndarray:
        samples, shade, weights = cell_samples(state, args, SPEC)
        colours = glow.sample_palette(palette, shade)
        colour_sum, density = glow.splat(width, height, samples, colours, weights)
        linear = glow.flame_map(colour_sum, density, reference, boost=SPEC["boost"])
        linear = glow.bloom(linear, threshold=args.bloom_threshold, strength=args.bloom_strength)
        return glow.compose(glow.to_bytes(glow.tone_map(linear, exposure=SPEC["exposure"])), caption)

    cover = draw(model)
    stem = f'{SPEC["slug"]}_{width}x{height}_{args.duration:g}s_{args.fps}fps'
    if args.hook:
        stem += "_hook_plex"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cover).save(args.output_dir / f"{stem}.cover.png")
    if args.preview:
        print(f"Saved {args.output_dir / (stem + '.cover.png')}")
        return

    output = args.output_dir / f"{stem}.mp4"
    encoder = start_encoder(output, width, height, args.fps)
    assert encoder.stdin is not None
    try:
        # Frame zero is the finished object, because frame zero is the grid
        # thumbnail. The run then starts over from nothing behind it.
        encoder.stdin.write(cover.tobytes())
        model = Phyllotaxis(height, width, radius=args.head_radius, primordia=organs)
        for index in range(1, frames):
            if index > 1:
                model.step(SPEC["steps_per_frame"])
            encoder.stdin.write(draw(model).tobytes())
            if index % 60 == 0:
                print(f"  frame {index}/{frames}", flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed.")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
