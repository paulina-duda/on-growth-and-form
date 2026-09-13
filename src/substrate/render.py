#!/usr/bin/env python3
"""Substrate: processes filmed on a medium, in the luminous house style.

This edition is about *where* a process happens. What it films is something a
microscope could be pointed at -- and the piece published here is the one that
is not alive at all: an active nematic, an organism's own parts with the
organism taken away and the motors left running.

Everything structural is inherited -- black field, additive accumulation into a
float buffer, log-density tone mapping, multi-scale bloom, spaced title,
monospace caption, a cover frame for the grid. Two things are not.

**Eight seconds, and the clip is a loop.** It opens on the finished organism,
cuts to a single seed, and grows back to exactly the frame it opened on. The
first thing anyone sees is the payoff, which is what buys the second of
attention; the cut is what turns that into a question; and because the last
frame is the first frame, the loop closes without a seam and the answer plays
again before anyone notices it has.

**Growth is paced by measurement, not by the clock.** Every one of these
processes accelerates or stalls on its own schedule -- a colony creeps and then
floods, an embryo doubles, a fractal's edge slows as the square root of its
material. Left on a linear timeline each one crawls and then bolts. So
each model reports a scalar for how far along it is, and frames are placed at
equal intervals of *that*, which is what keeps the growth hypnotic rather than
merely present. A travelling wave is the exception that proves the rule: it
moves at a fixed speed, so equal steps of the clock already are equal steps of
the process, and such a piece is banked one state per frame.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

import glow
import growths


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "out"
# The cells that have left the plane, deliberately off the ramp and warm against
# it: they are the one thing in the frame that is no longer being measured.
UPRIGHT = (255, 214, 122)
NEMATIC_UPRIGHT = (255, 252, 240)
# The mutants, off the ramp because they are not on the same axis as the lawn:
# they are a different lineage, and the piece is about the difference.
MUTANT = (255, 122, 96)
# Colour in `defect` is how recently a tear went past, and almost all of the
# drop has had no tear through it lately -- so, like AGAR, **no stop is dark**.
# Brightness here comes from how fast the film is moving, not from the ramp; a
# dark low end would black out the quiet film, which is most of every frame.
#
# **The low end is the palette.** Roughly nine pixels in ten sit at stops 0-2,
# so the film's colour is the whole impression the grid gives and the tear's is
# only ever a sparkle. Saturating the accent does nothing measurable; the first
# ice-and-ember draft moved mean saturation inside the drop from 0.272 to 0.381
# and still read as grey. Whatever this piece is going to look like on the feed
# has to be decided at stop 0.
#
# **Shipped.** Deep red film, cold tear, and the only one of the candidates
# that runs hot-to-cold rather than cold-to-hot -- so a tear reads as a cut
# rather than as a burn. Saturation inside the drop 0.662, against 0.272 for
# the steel it replaced. The cyan is blue used the way house rule 6 allows,
# as an accent of a few percent, and nothing else on the substrate grid is red.
# Green and blue are held near zero in the low stops on purpose: the tone curve
# and the bloom lift a dark red towards pink, and a pixel carrying both film
# and tear samples averages red against cyan into exactly that. The first pass
# at (120, 26, 38) came out rose and sat on CYTOSOL.
SCORCH = [(112, 8, 10), (168, 14, 14), (216, 34, 22), (60, 228, 244), (140, 244, 250), (232, 254, 255)]
# Steel film, ember tear. The original cut, kept as the quiet alternative --
# correct, and too grey to hold a place on the grid.
FILAMENT = [(44, 66, 86), (72, 104, 128), (134, 140, 140), (206, 160, 108), (252, 186, 96), (255, 246, 224)]
# The same ramp with a petrol film instead of a steel one. More saturated, and
# closer to the green PHOSPHOR and the cyan EPITHELIUM already on the grid,
# which is why it is not the default. Same rule -- no dark stop.
TEAR = [(30, 70, 72), (54, 108, 106), (118, 150, 138), (206, 160, 108), (252, 186, 96), (255, 246, 224)]
# Violet film, acid tear. The widest luminance gap of the candidates, so the
# accent is the one that genuinely reads as a second colour rather than as a
# sparkle. Not shipped -- SCORCH read better on the motion check -- but not
# rejected for sharing a family with the other candidates either: the same
# violet-to-magenta-to-cyan ramp already runs on published pieces elsewhere in
# the account, so a repeated family is not, on its own, a reason to drop a
# palette.
REAGENT = [(78, 40, 122), (120, 58, 180), (168, 108, 224), (198, 236, 76), (226, 250, 118), (250, 255, 212)]
# Glacier film, ember tear. At 0.602 saturation this is a whole blue piece,
# not blue as an accent -- there used to be a house rule against exactly that,
# since dropped. Kept as a measured alternative; SCORCH shipped instead for
# the hot-to-cold reading, not because QUENCH broke a rule.
QUENCH = [(16, 86, 124), (30, 132, 180), (96, 186, 216), (255, 116, 10), (255, 164, 40), (255, 234, 190)]

DEFECT_PALETTES = {
    "scorch": SCORCH,
    "filament": FILAMENT,
    "tear": TEAR,
    "reagent": REAGENT,
    "quench": QUENCH,
}

EDITIONS: dict[str, dict] = {
    "defect": {
        "kind": "points",
        "title": "Defect",
        "slug": "defect_active-nematic_substrate",
        "palette": SCORCH,
        "exposure": 1.12,
        "boost": 1.15,
        "caption": (
            "active nematic · kinesin on microtubules (Sanchez 2012)",
            "align · slide · bend · tear",
            "0 defects to 528 · each one born with its opposite",
        ),
        "hook": ("Nothing here is alive. It still cannot rest.",),
    },
}


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------


def build_overlay(width: int, height: int, spec: dict, args) -> Image.Image:
    """Title, hook and data block — one text layer, drawn once, held all clip.

    The hook sits in the strip between the form and the data block, centred,
    a few points above the block so it reads as the louder of the two and no
    louder than that: every pixel it takes is a pixel the organism gives up.
    Plex regular throughout, per house rule 5.

    `--no-text` returns an empty layer. That is what a T4 legibility still is
    rendered with: a title names the subject, and a still captioned with its own
    name asks "is this a good drawing of X" rather than "what is this".
    """
    if not args.text:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay = glow.make_caption(
        width,
        height,
        spec["title"],
        spec["caption"],
        equation_size=args.caption_size,
        margin=args.margin,
        top_margin=args.title_top,
        bottom_margin=args.caption_bottom,
        scrim=args.scrim,
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


def text_keepout(width: int, height: int, spec: dict, args) -> np.ndarray:
    """Ground the colony is not allowed onto: a ragged margin, and a halo of
    clear substrate around every glyph.

    The halo is taken from the text layer's own alpha, not from hand-measured
    boxes, so it follows the actual letterforms and stays correct if the copy or
    the type size changes. The scrim is switched off while measuring it -- it is
    a soft wash over most of the frame, and dilating that would forbid half the
    picture.

    **Both edges are perturbed by one smooth noise field, and that is not
    decoration.** A straight inset reads as a UI panel with the mycelium poured
    into it: the colony stops dead on a horizontal line, which no colony does.
    Letting the boundary wander by roughly its own depth costs nothing and the
    edge goes back to looking like the limit of a substrate.

    Cropping the render would slice filaments mid-stride. Forbidding the ground
    means the black is black because nothing was ever allowed to grow there, and
    because the model senses the mask, tips turn away from it rather than dying
    on it.
    """
    ink_only = copy.copy(args)
    ink_only.scrim = 0.0
    alpha = np.asarray(build_overlay(width, height, spec, ink_only).split()[-1])
    options = spec["keepout"]

    generator = np.random.default_rng(options.get("seed", 20260902))
    noise = ndimage.gaussian_filter(
        generator.normal(size=(height, width)).astype(np.float32), sigma=options.get("sigma", 90.0)
    )
    noise = noise / max(float(np.abs(noise).max()), 1e-9)

    # The bottom margin is not a number anyone chose -- it is measured off the
    # copy block itself, so the hook and the data block always sit on black and
    # stay there if the copy changes length. Everything above is a halo around
    # the title only, which reads as the word stamped into the mat.
    lower_ink = np.flatnonzero((alpha[height // 2 :] > 8).any(axis=1))
    copy_top = height // 2 + int(lower_ink[0]) if len(lower_ink) else height
    bottom = height - copy_top + options.get("copy_clearance", 34.0)

    rows, columns = np.indices((height, width))
    rough = options.get("roughness", 0.0) * noise
    forbidden = (
        (rows < options["top"] + rough)
        | (height - 1 - rows < bottom + rough)
        | (np.minimum(columns, width - 1 - columns) < options["side"] + rough)
        | (ndimage.distance_transform_edt(alpha <= 8)
           < options["clearance"] + options.get("halo_roughness", 0.0) * noise)
    )

    # Open the growable region by a disc, then drop what is left over as small
    # islands. Without this the mask leaves slivers -- between a halo and the
    # margin, or along the bottom edge where the noise thins it -- and a sliver
    # narrower than the sensor is a trap: tips inside it cannot smell a way out,
    # mill around, and pile density into a strip that tone-maps to solid white
    # directly under the data block. Measured on the first cut of this variant.
    # Two distance transforms rather than binary_opening with a 61 x 61
    # structure, which is the same result and far cheaper.
    reach = options.get("throat", 30.0)
    allowed = ~forbidden
    eroded = ndimage.distance_transform_edt(allowed) >= reach
    allowed = ndimage.distance_transform_edt(~eroded) <= reach
    labels, found = ndimage.label(allowed)
    if found:
        areas = ndimage.sum(allowed, labels, range(1, found + 1))
        survivors = 1 + np.flatnonzero(areas >= options.get("min_patch", 40_000))
        allowed = np.isin(labels, survivors)
    return ~allowed


def even_schedule(metric: np.ndarray, frames: int) -> np.ndarray:
    """State indices placed at equal intervals of progress, not of time.

    The metric is forced upward before inverting: a couple of these processes
    can dip -- cells drift during relaxation, hyphal tips fuse and stop -- and a
    non-monotonic curve inverts into a schedule that runs backwards for a frame.
    """
    curve = np.maximum.accumulate(np.asarray(metric, dtype=np.float64))
    curve -= curve[0]
    curve /= max(curve[-1], 1e-9)
    return np.searchsorted(curve, np.linspace(0.0, 1.0, frames)).clip(0, len(curve) - 1)


def place_square(field: np.ndarray, height: int, width: int, factor: int) -> np.ndarray:
    """Block-upsample a square simulation and centre it in the frame."""
    grown = np.repeat(np.repeat(field, factor, axis=0), factor, axis=1)
    canvas = np.zeros((height, width), dtype=np.float32)
    for axis, (target, source) in enumerate(zip((height, width), grown.shape)):
        if source > target:
            start = (source - target) // 2
            grown = np.take(grown, np.arange(start, start + target), axis=axis)
    offset_y = (height - grown.shape[0]) // 2
    offset_x = (width - grown.shape[1]) // 2
    canvas[offset_y : offset_y + grown.shape[0], offset_x : offset_x + grown.shape[1]] = grown
    return canvas


def place_band(field: np.ndarray, height: int, width: int, factor: int, top: int) -> np.ndarray:
    """Block-upsample a rectangular simulation into the slide band.

    The band is the *world*, not a crop of a bigger one -- the plate really is
    that shape, and nothing was ever simulated above or below it, which is why
    the black is black. `place_square` centres; this one seats the field at a
    given row so the band lands on the shipped slide geometry.
    """
    grown = np.repeat(np.repeat(field, factor, axis=0), factor, axis=1)
    canvas = np.zeros((height, width), dtype=np.float32)
    rows, columns = grown.shape
    offset_x = (width - columns) // 2
    canvas[top : top + rows, offset_x : offset_x + columns] = grown
    return canvas


def tone(colour_sum: np.ndarray, density: np.ndarray, reference: float, spec: dict, args) -> np.ndarray:
    """A piece once shipped on a non-default look whose filename never said so, and
    a re-render silently changed the cut. A piece that is not on the default look
    pins it here instead of relying on the flags being remembered."""
    linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
    linear = glow.bloom(
        linear,
        threshold=spec.get("bloom_threshold", args.bloom_threshold),
        strength=spec.get("bloom_strength", args.bloom_strength),
    )
    return glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"]))
def defect_timeline(spec: dict, args) -> tuple[Callable[[float], np.ndarray], np.ndarray]:
    height, width = args.height, args.width
    model = growths.Nematic(
        args.defect_size,
        activity=args.defect_activity,
        dt=args.defect_dt,
        afterglow=args.defect_afterglow,
    )
    # Two passes, and the reason is an old one: a scheduler can only repeat a
    # state or skip one. Pass one measures the progress curve;
    # pass two re-runs the same deterministic simulation and banks a state at
    # exactly the step each frame wants. The frames are then played straight
    # through, so there is nothing left for a scheduler to stutter over. Two
    # passes cost forty seconds against one banked run's twenty, and they also
    # cut the memory: 229 states instead of 701.
    def survey() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probe = growths.Nematic(
            args.defect_size, activity=args.defect_activity,
            dt=args.defect_dt, afterglow=args.defect_afterglow,
        )
        order, counts, speed = [], [], []
        for step in range(args.defect_steps + 1):
            if step % args.defect_stride == 0:
                order.append(probe.order())
                plus, minus = probe.defects()
                counts.append(plus + minus)
                speed.append(float(probe.arrays()[2][probe.dish].mean()))
            probe.step(1)
        return (np.asarray(order, dtype=np.float64),
                np.asarray(counts, dtype=np.float64),
                np.asarray(speed, dtype=np.float64))

    started = time.time()
    order, counts, speed = survey()

    # No single scalar paces this, and it takes three of them.
    #
    # For the first six thousand steps the film is whole: there are no defects
    # at all to count, and the only thing happening is the alignment buckling,
    # which only the order parameter sees. After it breaks, the order parameter
    # is pinned near zero for the rest of the clip and the tearing is all that
    # is left developing.
    #
    # The defect count alone will not pace that second half either, and this is
    # what the first cut got wrong: it is an integer, so it is a step function.
    # 54.6% of consecutive banked states hold the same count and the longest
    # identical run is 124 states -- 3,720 simulation steps of a curve that does
    # not move. Paced on it the clip measured **15.4% frozen after the hold**
    # against a house norm of 4%. The third term is the cumulative mean flow
    # speed: how far the film has slid in total, which is continuous, never
    # flat, and still a measurement of the process rather than of the clock.
    total = np.cumsum(speed)
    total -= total[0]
    progress = (
        (1.0 - order / max(float(order[0]), 1e-9))
        + args.defect_weight * (counts / max(float(counts.max()), 1.0))
        + args.defect_travel_weight * (total / max(float(total.max()), 1e-9))
    )
    curve = np.maximum.accumulate(progress)
    curve -= curve[0]
    curve /= max(float(curve[-1]), 1e-9)
    wanted = np.interp(
        np.linspace(0.0, 1.0, args.duration_frames),
        curve, np.arange(len(curve), dtype=np.float64) * args.defect_stride,
    )
    wanted = np.round(wanted).astype(np.int64)
    # Strictly increasing: two frames on one simulation step would be the very
    # stutter the two passes are here to remove.
    for index in range(1, len(wanted)):
        wanted[index] = max(wanted[index], wanted[index - 1] + 1)
    wanted = np.minimum(wanted, args.defect_steps)

    model = growths.Nematic(
        args.defect_size, activity=args.defect_activity,
        dt=args.defect_dt, afterglow=args.defect_afterglow,
    )
    states: list[tuple] = []
    cursor = 0
    for step in range(args.defect_steps + 1):
        while cursor < len(wanted) and wanted[cursor] == step:
            states.append(model.state())
            cursor += 1
        model.step(1)
    while len(states) < len(wanted):
        states.append(model.state())

    plus, minus = model.defects()
    print(
        f"  defect: {args.defect_steps:,} steps twice on {model.device} in {time.time()-started:.0f}s, "
        f"{len(states)} states banked one per frame, drop radius {model.radius:.1f} of "
        f"{args.defect_size}, {plus + minus} defects (+{plus}/-{minus}), "
        f"order {order[0]:.3f} to {order[-1]:.4f}, film breaks at frame "
        f"{int(np.argmax(np.interp(wanted, np.arange(len(counts)) * args.defect_stride, counts) > 10))}"
        f" of {args.duration_frames}",
        flush=True,
    )
    schedule = np.arange(len(states))

    palette = glow.build_palette(DEFECT_PALETTES[args.defect_palette])
    caption = build_overlay(width, height, spec, args)
    scale = (min(width, height) * 0.44) / model.radius
    centre = (width * 0.5, height * 0.5)

    # Brightness is how fast this patch of film is moving, against one fixed
    # reference for the whole clip -- so "brighter" means the same thing in
    # every frame. Ranked per frame it would say nothing, because the contrast
    # between the fast and slow parts barely changes; what changes is the
    # overall speed, and that is the measurement worth keeping.
    sampled = np.concatenate([
        states[index][2][model.dish].astype(np.float32)
        for index in range(0, len(states), max(len(states) // 24, 1))
    ])
    speed_reference = float(np.percentile(sampled, args.defect_speed_percentile))
    print(f"  defect: speed reference {speed_reference:.4f} "
          f"(p{args.defect_speed_percentile:g} over the banked run)", flush=True)
    reference = 1.0

    def buffers(index: int) -> tuple[np.ndarray, np.ndarray]:
        # The same seed every frame, deliberately. Re-scattering the seeds each
        # time makes the strokes crawl, and the crawl is louder than the film.
        points, values, weights = model.samples(
            states[index], scale, centre,
            seeds=args.defect_seeds, walk=args.defect_walk,
            generator=np.random.default_rng(args.defect_seed),
        )
        shade = np.clip(weights / speed_reference, 0.0, 1.0)
        colours = glow.sample_palette(palette, np.clip(values, 0.0, 1.0))
        # The floor is the film itself. An aligned nematic generates no flow at
        # all -- no bend, no force, no motion -- so speed alone draws the first
        # state of the clip as an empty frame, and the clip would open on black
        # and fade the drop in. A microscope would see the filaments whether or
        # not they were moving; the floor is that, and the speed on top of it is
        # what the activity adds.
        weight = args.defect_floor + (1.0 - args.defect_floor) * shade ** args.defect_gamma
        return glow.splat(width, height, points, colours, weight.astype(np.float32))

    def draw(u: float) -> np.ndarray:
        index = int(schedule[min(int(u * (len(schedule) - 1)), len(schedule) - 1)])
        colour_sum, density = buffers(index)
        return glow.compose(tone(colour_sum, density, reference, spec, args), caption)

    _, final_density = buffers(len(states) - 1)
    reference = float(np.percentile(final_density[final_density > 0], 92.0))
    return draw, draw(1.0)


TIMELINES = {
    "defect": defect_timeline,
}


def render_edition(name: str, args: argparse.Namespace) -> Path:
    spec = EDITIONS[name]
    draw, finished = TIMELINES[name](spec, args)

    stem = f"{spec['slug']}_{args.width}x{args.height}_{args.duration:g}s_{args.fps}fps"
    if spec.get("look"):
        stem += f"_{spec['look']}"
    if args.hook and spec.get("hook"):
        # The suffix every re-cut into the current layout carries, so a hooked Plex
        # cut and an older one sit side by side in the folder without ambiguity.
        stem += "_hook_plex"
    if args.tag:
        # How a re-cut lands alongside the version it descends from instead of
        # on top of it. Every renderer in the account shares this mechanism.
        stem += f"_{args.tag}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(finished).save(args.output_dir / f"{stem}.cover.png")
    if args.preview:
        return args.output_dir / f"{stem}.cover.png"

    output = args.output_dir / f"{stem}.mp4"
    encoder = start_encoder(output, args.width, args.height, args.fps)
    assert encoder.stdin is not None
    try:
        payload = finished.tobytes()
        for _ in range(args.hold):
            encoder.stdin.write(payload)
        growing = args.duration_frames - args.hold
        for index in range(growing):
            # Ends on u = 1, which is the frame the clip opened on, so the loop
            # has no seam in it.
            encoder.stdin.write(draw((index + 1) / growing).tobytes())
            if (index + 1) % 60 == 0:
                print(f"  {name}: frame {index + 1}/{growing}", flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while rendering {name}.")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=sorted(EDITIONS), action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview", action="store_true", help="Save the cover still and stop.")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--duration", type=float, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold", type=int, default=11, help="frames held on the finished form")
    parser.add_argument("--margin", type=int, default=64, help="left inset, shared by all three text layers")
    # Not symmetric and not negotiable: the Reel player lays its header over the
    # top of the frame and the account row over the bottom, and the bottom needs
    # the most clearance of the two.
    parser.add_argument("--title-top", type=int, default=240)
    parser.add_argument("--caption-bottom", type=int, default=190)
    parser.add_argument("--caption-size", type=int, default=27)
    parser.add_argument("--scrim", type=float, default=0.95)
    parser.add_argument("--no-hook", dest="hook", action="store_false", help="drop the hook line")
    parser.add_argument("--hook-size", type=int, default=34)
    parser.add_argument("--hook-gap", type=int, default=82, help="hook ink down to the data block's ink")
    parser.add_argument("--bloom-threshold", type=float, default=0.30)
    parser.add_argument("--bloom-strength", type=float, default=0.60)
    parser.add_argument("--tag", help="suffix appended to the filename, e.g. v2")
    parser.add_argument("--no-text", dest="text", action="store_false",
                        help="render the form with no title, hook or data block — a T4 legibility still")
    parser.add_argument("--fuse-warm", type=float, default=22.0,
                        help="steps a segment takes to warm from slate to ember once its tip fuses")
    parser.add_argument("--tip-boost", type=float, default=2.2)
    parser.add_argument("--tip-decay", type=float, default=26.0)
    parser.add_argument("--divide-rate", type=float, default=0.035)
    parser.add_argument("--pile-size", type=int, default=379)
    parser.add_argument("--pile-factor", type=int, default=3)
    parser.add_argument("--grains", type=int, default=150_000)
    parser.add_argument("--front-boost", type=float, default=0.55)
    parser.add_argument("--epsilon", type=float, default=0.05, help="how brief the excited state is")
    parser.add_argument("--roughness", type=float, default=0.016, help="spread of the excitability field")
    parser.add_argument("--afterglow", type=float, default=45.0, help="half-life of the phosphor, in steps")
    parser.add_argument("--neurons", type=int, default=12_000)
    parser.add_argument("--cell-samples", type=int, default=48, help="splat samples per neuron")
    parser.add_argument("--cell-radius", type=float, default=6.5, help="radius of one neuron, in pixels")
    # Zero, and pinned here rather than left on the command line: the cut that
    # was judged was rendered with the trace off. At 0.20 and above the slow
    # layer lights every cell that has fired in the last second, which by the
    # end of the clip is all of them, and the frame fills with an even carpet
    # that buries the structure the layer was meant to sit under. An earlier
    # piece shipped a look its filename did not record, and re-rendering it from
    # the name silently changed the cut; this does not repeat that.
    parser.add_argument("--cell-trace", type=float, default=0.0,
                        help="how bright tissue stays after the spike has gone")
    parser.add_argument("--mutation", type=float, default=0.0022, help="chance a division founds a lineage")
    parser.add_argument("--beneficial", type=float, default=0.16, help="fraction of those that divide faster")
    parser.add_argument("--advantage", type=float, default=0.10, help="how much faster")
    # Area, not radius, and this was decided by measurement rather than taste.
    # Paced by radius the front advances at a steady speed and the change
    # profile comes out comet-shaped -- 9.5% of the growth in the first quarter
    # and 40.4% in the last -- but the colony spends the opening two seconds as
    # a dot, and a dot advancing one cell of radius per frame changes almost no
    # pixels. That cut measured 22.6% frozen frames with 52 of the 54 inside the
    # first two seconds, which is the worst place a freeze can land. Equal area
    # per frame is equal newly-lit pixels per frame, which is the thing the
    # frozen-frame test actually measures: 5.4%, and 1.3% once the cover hold is
    # discounted. The intermediate, area^0.75, splits the difference at 7.9%.
    parser.add_argument("--pacing", type=float, default=1.0, help="0.5 paces by radius, 1.0 by area")
    parser.add_argument("--inoculum", type=float, default=22.0, help="radius of the drop, in lattice cells")
    parser.add_argument("--hue-step", type=float, default=0.26, help="how far a mutation moves along the ramp")
    parser.add_argument("--tip-boost-front", type=float, default=1.35)
    parser.add_argument("--tip-decay-front", type=float, default=26.0)
    parser.add_argument("--landings", type=int, default=130)
    parser.add_argument("--landing-span", type=int, default=560, help="steps over which phage arrive")
    # The latent period sets how wide the bursting ring is, and the ring is
    # the only thing on the plate that moves. At 1.2 it is two cells across
    # and the clip measured 32.6% frozen.
    parser.add_argument("--latent", type=float, default=1.2)
    parser.add_argument("--phage-decay", type=float, default=0.04)
    parser.add_argument("--resistant-rate", type=float, default=0.30)
    parser.add_argument("--adsorption", type=float, default=9.0)
    parser.add_argument("--lawn-start", type=float, default=0.10)
    parser.add_argument("--colony-weight", type=float, default=0.70)
    parser.add_argument("--fill-weight", type=float, default=0.60)
    parser.add_argument("--landing-delay", type=int, default=170, help="steps before the first phage lands")
    parser.add_argument("--plated", type=int, default=700_000, help="individual cells drawn on the plate")
    parser.add_argument("--mutant-boost", type=float, default=1.35)
    # `defect`. The drop is 0.44 of the grid, so the simulation square maps one
    # for one onto the frame width and the dish lands where the reel skill puts
    # it without any cropping.
    parser.add_argument("--defect-size", type=int, default=320)
    # Measured in the dish, at 0.86 ms a step on the GPU: 0 / 2 / 158 / 358 / 528
    # defects across 21,000 steps. Longer is denser and the last quarter gets
    # thinner -- 16% at 26,000 against 32% here.
    parser.add_argument("--defect-steps", type=int, default=21_000)
    parser.add_argument("--defect-stride", type=int, default=30, help="steps between banked states")
    parser.add_argument("--defect-activity", type=float, default=0.030, help="extensile active stress")
    parser.add_argument("--defect-dt", type=float, default=0.05)
    parser.add_argument("--defect-afterglow", type=float, default=400.0, help="half-life of the tear phosphor, in steps")
    parser.add_argument("--defect-seeds", type=int, default=320_000, help="streamlines drawn per frame")
    parser.add_argument("--defect-walk", type=int, default=9, help="samples along each streamline")
    parser.add_argument("--defect-seed", type=int, default=4, help="held fixed across frames on purpose")
    # 1.5 leaves the drop a flat milky wash: the log-density map compresses what
    # is already a narrow range and nothing in the film separates. 2.6 puts the
    # slow channels back into the dark, which is where the tearing shows.
    parser.add_argument("--defect-gamma", type=float, default=2.6, help="how hard speed is turned into brightness")
    parser.add_argument("--defect-speed-percentile", type=float, default=97.0)
    # Small on purpose. The floor buys the ordered film at the head of the clip;
    # paid past about 0.06 it also buys back the contrast the tearing is made of,
    # because the log-density map has only so much range. At 0.03 the still film
    # reads as a flat grey disc and the turbulence keeps its dark channels.
    parser.add_argument("--defect-floor", type=float, default=0.03, help="what the film is worth when it is not moving")
    # The film is whole for the first ~6,000 steps of 21,000. At 2.0 that phase
    # gets about a third of the clip, which is what it needs to read as a fabric
    # before it is a wreck.
    parser.add_argument("--defect-weight", type=float, default=2.0, help="weight on the defect count in the pacing")
    # The continuous term. Without it the pacing rides an integer step
    # function and the cut measured 15.4% frozen after the hold.
    parser.add_argument("--defect-travel-weight", type=float, default=1.0, help="weight on how far the film has slid in total")
    parser.add_argument("--defect-palette", choices=tuple(DEFECT_PALETTES), default="scorch")
    parser.add_argument("--upright-boost", type=float, default=1.5)
    parser.add_argument("--epsilon-ch", type=float, default=1.0, help="interface width, and the stability limit")
    parser.add_argument("--mixture", type=float, default=-0.35, help="negative makes the dense phase a minority")
    parser.add_argument("--droplet-reference", type=float, default=20.0, help="radius, in cells, that reads as white")
    parser.add_argument(
        "--stimulus", type=float, action="append", default=None,
        help="fractions of the clip at which a premature beat is delivered",
    )
    parser.add_argument("--exposure", type=float, help="override the edition's exposure")
    parser.add_argument("--boost", type=float, help="override the edition's boost")
    args = parser.parse_args()
    args.duration_frames = round(args.duration * args.fps)
    if args.stimulus is None:
        # The first beat goes out into clear tissue; these land in the wake of
        # the one before, which is the only place a wave can be broken.
        args.stimulus = [0.16, 0.34, 0.52, 0.70]
    return args


if __name__ == "__main__":
    options = parse_args()
    for edition_name in options.edition or list(EDITIONS):
        for key in ("exposure", "boost"):
            if getattr(options, key) is not None:
                EDITIONS[edition_name][key] = getattr(options, key)
        print(f"Rendering {edition_name} ...", flush=True)
        print(f"Saved {render_edition(edition_name, options)}", flush=True)
