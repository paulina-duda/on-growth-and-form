#!/usr/bin/env python3
"""Three morphogenesis algorithms, rendered as wetware.

Same house style as the attractor and protein pieces -- black field, additive
accumulation, multi-scale bloom, log-density tone mapping, spaced title,
monospace caption, a cover frame of the finished form for the grid -- but the
palettes move to the cyan-and-magenta end, and what the pieces are *of* is a
process rather than an object.

Each clip opens on the finished form and then plays its growth from the first
step. Growth is the only motion: these are plane processes, and turning them
would be a camera move pasted onto something that does not have a far side.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scipy.spatial import cKDTree

import glow
import morphogens


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "out"

# Violet through magenta into a warm white. Deliberately not `CULTURE`, which
# is the green the other growing piece already owns, and not `SYNAPSE`, which
# ends cold -- an apex is the warm end of this account, not the cold one.
MERISTEM = [(8, 0, 26), (72, 0, 140), (190, 0, 170), (255, 70, 150), (255, 170, 120), (255, 240, 210)]
# Shipped on phyllotaxis 2026-08-31, replacing MERISTEM. Same
# violet-to-magenta arms -- the account's colour vocabulary for age
# doesn't change -- but the last two stops swap the warm coral/cream core
# for a cyan the account already uses elsewhere, so the
# youngest primordia at the centre pop as a complementary accent against
# the magenta arms instead of blending into them. Only the brightest few
# percent of the ramp is affected, since colour is age and the newest
# organs are the smallest ring.
APEX = [(8, 0, 26), (72, 0, 140), (190, 0, 170), (255, 60, 190), (80, 230, 255), (215, 255, 255)]

# Two pigment cells, so two palettes that sum where a stripe border falls --
# the same two-channel answer other pieces use. The split is the
# whole subject here: which of the two kinds of cell won that patch of skin.
# Cool for the melanophore, warm for the xanthophore, and the melanophore ramp
# is taken through violet rather than straight up the blues, because half the
# disc is this colour and a blue that large stops being an accent.
#
# The low stop of each ramp is doing more work here than in any other edition,
# and it is worth saying why. Shade is recency, and a cell that has settled
# sits at the very bottom of its ramp -- so **the first stop is the colour of a
# stripe's interior**, which is most of the frame, and the upper stops only
# appear along a border that is currently being argued over. Both ramps
# therefore start at a real colour rather than near-black: at (28,8,0) the
# xanthophore interiors came out a desaturated brown, measured at blue/red 0.42
# against 0.35 once the stop was opened up.
#
# The two are deliberately NOT matched in luminance -- the violet measures 60
# against the gold's 109. A melanophore is a black cell and a xanthophore a
# yellow one, so the dark band belongs darker; balancing them would have meant
# pushing the melanophore toward magenta to buy luminance blue cannot carry,
# and that is fighting the animal to satisfy a number.
MELANO = [(30, 0, 86), (110, 0, 205), (196, 80, 250), (240, 200, 255)]
XANTHO = [(92, 38, 0), (196, 86, 0), (255, 170, 28), (255, 238, 180)]

# `tear`. Colour is how stretched a cell is, scaled against the finished
# frame's 98th percentile rather than ranked -- most of the tissue sits near
# zero strain (median 0.03 against 0.13 at the 98th, measured), and ranking
# would drag the relaxed sheet up the ramp. So the relaxed tissue lands about a
# quarter of the way up, in indigo-violet, and a line the animal is about to
# open along runs up through a hot red-pink into orange: stretch reads as heat.
# Warm top, cold bottom, and deliberately not the plum-to-gold or
# magenta-to-cyan ramps other pieces already own.
FAULT = [(14, 4, 44), (44, 14, 104), (112, 22, 146), (240, 48, 110), (255, 142, 44), (255, 236, 160)]

# `tear`'s rework. Colour is which animal a cell belongs to, so the ramp is
# cyclic -- first stop and last stop are the same -- and a tear hands the
# smaller half a tint a third of the way round it. Two halves of one body come
# out clearly different and no body is a special case, which is what makes the
# event read. Brightness still carries stretch, so a line about to give still
# lights up; hue says whose it is.
KIN = [(230, 70, 140), (235, 150, 60), (110, 220, 150), (70, 150, 230), (160, 80, 220), (230, 70, 140)]

# The two neon cuts of `tear`, on Paulina's ask. Same job as KIN -- a cyclic
# ramp, identity by tint -- and both measure about 0.78 mean saturation over
# the lit pixels of frame one against KIN's 0.552. PRISM spreads the hues as
# far apart as they go, so no two bodies are mistakable; it is also the loud
# one, and its lime against magenta is a pairing an earlier piece died on. UV
# stays in the blacklight band the account already uses, at the cost of cyan
# and electric blue being the closest pair on the ramp.
PRISM = [(255, 0, 170), (255, 130, 0), (190, 255, 0), (0, 255, 190), (90, 90, 255), (255, 0, 170)]
UV = [(255, 0, 200), (175, 60, 255), (60, 140, 255), (0, 240, 230), (255, 60, 150), (255, 0, 200)]

EDITIONS: dict[str, dict] = {
    "phyllotaxis": {
        "kind": "spiral",
        "title": "Phyllotaxis",
        "slug": "phyllotaxis_primordia_apex",
        # APEX, not MERISTEM -- chosen 2026-08-31 over a side-by-side
        # render of both. Same violet-to-magenta arms; the youngest few
        # percent of the ramp (the newest primordia, the smallest ring, right
        # at the centre) swap the old warm coral/cream for the cyan the
        # account already uses elsewhere, so the core reads as a
        # complementary accent against the arms instead of blending into them.
        "palette": APEX,
        "exposure": 1.16,
        "boost": 1.24,
        "steps_per_frame": 5,
        "settle": 0,
        "cell_radius": 9.0,
        "cell_samples": 48,
        "caption": (
            "shoot apical meristem  ·  Douady & Couder 1992",
            "grow · inhibit · place the next one",
            "spiral counts 8, 13, 21 out to 21, 34, 55",
            "divergence 137.4°  ·  no angle is in the rule",
        ),
        "hook": ("The plant is not counting. You are.",),
    },
    "stripe": {
        "kind": "skin",
        "title": "Stripe",
        "slug": "stripe_pigment-cells_skin",
        "palette": MELANO,
        "palette_b": XANTHO,
        "exposure": 1.16,
        "boost": 1.22,
        # Eight steps a frame rather than four. The reaction has to reinsert a
        # stripe faster than the skin carries the neighbours apart, and at four
        # it loses that race: the stripes just get fatter.
        "steps_per_frame": 8,
        # Frame one is a patterned disc, not the salt-and-pepper the cells
        # start in. Without this the thumbnail is grey noise.
        "settle": 260,
        "cell_radius": 1.9,
        "cell_samples": 6,
        "look": "bloom",
        "caption": (
            "zebrafish pigment pattern  ·  Nakamasu 2009",
            "support close · suppress far · switch",
            "33,000 cells become 118,000 · 667,000 switches",
        ),
        # Shipped 2026-09-05 on the data block's own fourth line rather than
        # the Turing-reversal line ("Turing predicted chemicals. These are
        # cells.") it replaces -- that line is dropped from the block below so
        # the fact isn't stated twice.
        #
        # One line, and that is a clearance constraint rather than a taste one.
        # A dish is centred with radius 0.44 x the short side, which puts the
        # disc's bottom at row 1435; a two-line hook starts at row 1420 and
        # overlaps it by 15 px before bloom. Every shipped dish hook is one
        # line for this reason.
        "hook": ("The stripe's width is fixed; the skin's is not.",),
    },
    # The same rework seeded evenly instead of at random. Measured over the
    # clip, a centred disc 340 px across is lit 10.8% on average in the random
    # cut and under 15% in 205 frames of 300 -- two thirds of the clip with an
    # empty middle, which is what Paulina saw. Six animals on a low-discrepancy
    # spread: 36.2% mean, never below 26.4%, no frame under 15%. Tears
    # 0 / 5 / 12 / 17 / 29, 17.2% first quarter and 41.4% last.
    "tear3": {
        "kind": "placozoa",
        "title": "Tear",
        "slug": "tear_placozoan-spread_kin",
        "palette": KIN,
        "look": "bloom",
        "exposure": 1.10,
        "boost": 1.12,
        "steps_per_frame": 20,
        "settle": 2000,
        "model": {"world": "field", "box": (108.0, 192.0), "animals": 6, "size": 10.0,
                  "v0": 0.45, "grow": 0.004, "spread": 2, "layout": "spread"},
        "cover": "first",
        "reference": "last",
        "cell_radius": 2.2,
        "cell_samples": 5,
        "floor": 0.0,
        "colour_reference": 100.0,
        "caption": (
            "Trichoplax adhaerens  ·  fission",
            "walk · turn to the pull · stretch · give",
            "motility-induced fracture, Prakash 2021",
        ),
        "hook": ("It reproduces by disagreeing with itself.",),
    },
}


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


def upsample(field: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour block upsample for fields simulated below frame size."""
    if field.shape == (height, width):
        return field
    rows = height // field.shape[0]
    columns = width // field.shape[1]
    return np.repeat(np.repeat(field, rows, axis=0), columns, axis=1)[:height, :width]


def compose_field(
    channels: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    reference: float,
    spec: dict,
    args: argparse.Namespace,
    caption: Image.Image,
) -> np.ndarray:
    """Tone map one or more scalar fields through their palettes.

    Each channel is a (density, shade, palette) triple. Keeping density and
    shade separate is the point: brightness should follow how much process is
    there, while hue can follow something else entirely -- in a reaction-diffusion
    piece it follows when each cell first lit, turning the colony into growth rings.

    The pipeline is otherwise the attractors' pipeline with the splat step
    removed: a field is already an accumulation buffer, so it goes straight into
    the same log-density map. Summing the channels is what lets two populations
    blend where they overlap instead of one painting over the other.
    """
    colour_sum = None
    density = None
    for field, shade, palette in channels:
        colour = glow.sample_palette(palette, np.clip(shade, 0.0, 1.0).astype(np.float32))
        weighted = colour * field[:, :, None]
        colour_sum = weighted if colour_sum is None else colour_sum + weighted
        density = field if density is None else density + field
    linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
    linear = glow.bloom(linear, threshold=args.bloom_threshold, strength=args.bloom_strength)
    frame = glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"]))
    return glow.compose(frame, caption)


def build_overlay(width: int, height: int, spec: dict, args) -> Image.Image:
    """Title, hook and data block -- the layout the substrate set settled on.

    Plex throughout, spaced bold title 240 px down, data block 190 px up, hook
    centred at 34 px with its lowest ink 82 px clear of the block, soft scrim at
    both edges. Reused rather than re-derived: the numbers were measured off the
    first cut to use it, pixel by pixel, and there is no reason for a second opinion.
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
        # veil, and a piece whose column runs to the top edge pays for it: so
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


def field_channels(model, spec, palette, palette_b, height, width, reference):
    """Build the (density, shade, palette) channels for a field edition."""
    if spec["kind"] in ("physarum", "bone"):
        # Two populations sharing a frame: same problem, same answer. Each carries its own density and its own palette, and the sum
        # is what lets a crossing read as a crossing instead of one of them
        # painting over the other. Shade has to come from the channel's own
        # density -- taking it from the total sends both palettes to their white
        # ends at once and throws the hue away.
        a, b = model.field()
        a, b = upsample(a, height, width), upsample(b, height, width)
        scale = 1.0 / max(reference or 1.0, 1e-9)
        return [(a, a * scale, palette), (b, b * scale, palette_b)]
    field = upsample(model.field(), height, width)
    return [(field, upsample(model.growth_rings(), height, width), palette)]


def curve_samples(points: np.ndarray, target: float = 0.6) -> np.ndarray:
    """Subdivide a closed polyline so successive samples nearly touch.

    Without this the curve is drawn as a string of dots wherever it runs fast
    across the frame -- the same problem the attractor traces had, and the same
    fix.
    """
    following = np.roll(points, -1, axis=0)
    delta = following - points
    length = np.linalg.norm(delta, axis=1)
    counts = np.clip(np.ceil(length / target), 1, 32).astype(np.int64)
    total = int(counts.sum())
    segment = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    starts = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    fraction = ((np.arange(total, dtype=np.int64) - starts[segment]) / counts[segment]).astype(np.float32)
    return (points[segment] + fraction[:, None] * delta[segment]).astype(np.float32), segment


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

    A tissue cell is a few square microns and a floret is an organ, so the two
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


def sheet_samples(model, args, spec=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the sheet as a tiling, and light it at the junctions.

    Two earlier drawings of this are why it works this way.

    Splatting each cell as a blob of its own radius gave loose speckle at 40
    samples a cell -- one per 80 px^2 inside a 32 px radius, 10.6% lit -- and
    filling the blob properly gave the next failure instead: a cloud of
    separate round discs with black between them, because a cell that is 34%
    into its contraction is well inside the radius it packs at. Tissue does not
    open gaps when a cell contracts; its neighbours take up the slack.

    So the sheet is sampled rather than the cells. The frame is walked on a
    fixed grid, each sample is assigned to the cell whose *power* distance is
    smallest -- ``|x - p|^2 - r^2``, which is the tiling a set of unequal discs
    actually makes, and the plain nearest-cell rule puts the boundary halfway
    between a 32 px cell and an 18 px one, which is nowhere near where it is --
    and the sample is lit by how close it is to the junction between its cell
    and the next. That is honest as well as legible: the junction is where the
    actin belt is, so brightness is once again where the stuff is.

    The grid is fixed frame to frame, so nothing boils.
    """
    spec = spec or {}
    points, rate = model.cells()
    radii = model.radii().astype(np.float32)
    spacing = float(spec.get("sample_spacing", 2.4))
    width, height = float(model.w), float(model.h)

    xs = np.arange(spacing * 0.5, width, spacing, dtype=np.float32)
    ys = np.arange(spacing * 0.5, height, spacing, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    # Jittered once, from a fixed seed. A bare lattice at this pitch prints a
    # visible halftone screen over the whole frame -- it is the first thing you
    # see in the unjittered cut -- and jittering per frame instead would make
    # the sheet boil. Fixed offsets do neither.
    grid = grid + np.random.default_rng(13).uniform(
        -0.5, 0.5, size=grid.shape).astype(np.float32) * spacing

    k = min(8, len(points))
    d, idx = cKDTree(points).query(grid, k=k)
    # Power (Laguerre) distance: the tiling unequal discs really make.
    power = d.astype(np.float32) ** 2 - (radii[idx] ** 2)
    order = np.argsort(power, axis=1)
    first = order[:, 0]
    rows = np.arange(len(grid))
    own = idx[rows, first]
    p0 = power[rows, first]
    p1 = power[rows, order[:, 1]] if k > 1 else p0 + 1e6

    # Distance from the junction, in px. The power difference grows as roughly
    # twice the separation of the two centres per px of travel, so divide it
    # out rather than tuning a width against the cell size.
    sep = np.maximum(np.linalg.norm(points[own] - points[idx[rows, order[:, 1]]], axis=1), 1e-3)
    edge = np.maximum(p1 - p0, 0.0) / (2.0 * sep)

    junction = float(spec.get("junction_width", 5.0))
    floor = float(spec.get("interior", 0.16))
    # Medial myosin. A pulse in an amnioserosa cell flares in the middle of the
    # apical surface, not at the junctions -- it is a distinct pool from the
    # junctional belt and it is what the ratchet actually runs on. With this at
    # zero a pulsing cell is a bright outline; turned up, the cell fills as it
    # pulls, so the beat is a change of *texture* as well as of hue.
    #
    # Driven by the same continuous scalar, never by a threshold, and that is
    # not a stylistic preference. Measured on the beat: a hard cut at 0.70 of
    # the reference puts 352 highlight transitions a second in the frame with a
    # tenth of them lasting four frames or fewer, and at 0.85 it is 179 a
    # second with a p10 dwell of two frames. That is `culture`'s rejection --
    # blinking dots -- arrived at on purpose. Continuous, a cell takes about
    # nine frames to cross the ramp, so it swells instead of blinking.
    medial = float(spec.get("medial", 0.0))
    glow_edge = np.exp(-(edge / junction) ** 2)
    if medial > 0.0:
        # The exponent is what keeps this from swallowing the tiling. At a
        # linear response and medial 0.80 every amnioserosa cell fills, the
        # junction network inside the lens disappears and the frame goes back
        # to being a field of pastel blobs -- the thing the power diagram was
        # brought in to fix. Cubed, only the cells actually near peak fill.
        strength = np.clip(rate[own] / max(float(spec.get("_cref", 1.0)), 1e-9), 0.0, 1.0)
        glow_edge = np.maximum(glow_edge, medial * strength ** float(spec.get("medial_gamma", 3.0)))
    lit = floor + (1.0 - floor) * glow_edge
    # Brightness is cell height, so it is still "how much stuff is there" --
    # a contracting cell keeps its volume and gets taller. Without this the
    # amnioserosa and the epidermis sit at the same brightness and the hole
    # does not read at 200 px: the whole frame is one violet honeycomb.
    # No cutoff. The sheet is confluent: cells tile the plane and a cell that
    # constricts hands territory to its neighbours rather than opening a gap.
    # Cutting samples at 1.7x the drawn radius punched the contracted cells
    # out of the tissue and left the last second of the clip as scattered
    # discs in a black void, which is not what a closing epithelium does.
    weights = (lit * model.heights()[own]).astype(np.float32)
    return grid, rate[own], weights


def trail_samples(start: np.ndarray, end: np.ndarray, target: float = 0.6):
    """Sample each trail along its own length, and say where on it each sample fell.

    `tree_samples` does the sampling and throws the fraction away; a trail needs
    it, because the weight has to fall off towards the tail. Sampled by length
    rather than a fixed count per segment, or the long trails come out as dotted
    rules and it reads as a layout bug.
    """
    delta = end - start
    length = np.linalg.norm(delta, axis=1)
    counts = np.clip(np.ceil(length / target), 1, 48).astype(np.int64)
    total = int(counts.sum())
    segment = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    fraction = ((np.arange(total, dtype=np.int64) - offsets[segment]) / counts[segment]).astype(np.float32)
    return (start[segment] + fraction[:, None] * delta[segment]).astype(np.float32), segment, fraction


def swarm_samples(model, spec, colour_reference: float):
    """Draw a lawn of amoebae as the paths they have just walked.

    The first cut splatted every cell as a round blob on a flat weight floor,
    and 150,000 blobs on a floor is a carpet of grain: the streams read as
    texture rather than as anything going anywhere. Drawing each cell's recent
    path instead fixes both ends of that at once -- a recruited cell becomes a
    short luminous streak pointing where it is going, and a cell that has not
    moved collapses to the point it always was. That difference is the picture,
    and it is also the honest difference between a recruited amoeba and one
    still sitting where it starved.

    Three quantities, three jobs, unchanged: position places the sample, the
    phosphor is the weight so a wave crossing the plate reads as a wave, and
    hue is how far the signal has carried the cell.
    """
    path, travel, lit = model.trails()
    legs, cells = path.shape[0] - 1, path.shape[1]
    floor = spec.get("floor", 0.12)
    weight = (floor + (1.0 - floor) * lit).astype(np.float32)
    # Scaled, not ranked: the distribution is only mildly skewed (0.88) and
    # ranking would spread a full ramp across a lawn that has not moved yet,
    # which is a rainbow of noise on the frames that matter most. Referenced to
    # the finished state so the colour has the same arc the process does.
    shade = np.clip(travel / max(colour_reference, 1e-6), 0.0, 1.0).astype(np.float32)

    # Every cell gets a head, so nothing disappears on the frame it happens to
    # wrap and the lawn stays a lawn of individuals rather than of trails.
    radius, per_head = spec.get("cell_radius", 1.15), spec.get("cell_samples", 3)
    index = (np.arange(cells, dtype=np.int64)[:, None] * per_head
             + np.arange(per_head, dtype=np.int64)[None, :]) % len(DISC)
    samples = [(path[-1][:, None, :] + DISC[index] * radius).reshape(-1, 2)]
    shades = [np.repeat(shade, per_head)]
    weights = [np.repeat(weight / per_head, per_head)]

    # Then the path, leg by leg. A leg shorter than half a pixel is a cell that
    # was not going anywhere over that stretch and is already covered by the
    # head; a leg longer than the ceiling is a wrap, and the line between a
    # cell's two sides of the frame is a stripe nothing walked.
    ceiling = spec.get("trail_ceiling", 40.0)
    taper, gain = spec.get("trail_taper", 1.4), spec.get("trail_weight", 0.62)
    for leg in range(legs):
        a, b = path[leg], path[leg + 1]
        span = np.linalg.norm(b - a, axis=1)
        keep = (span > 0.5) & (span < ceiling)
        if not keep.any():
            continue
        trail, segment, fraction = trail_samples(a[keep], b[keep])
        # Where this sample sits along the whole path, 0 at the oldest point
        # and 1 at the cell. The streak is faintest where the cell has been and
        # brightest where it is, so it reads as motion rather than as a dash.
        along = ((leg + fraction) / legs).astype(np.float32)
        samples.append(trail)
        shades.append(shade[keep][segment])
        weights.append((weight[keep][segment] * gain * along ** taper).astype(np.float32))

    return (
        np.concatenate(samples).astype(np.float32),
        np.concatenate(shades).astype(np.float32),
        np.concatenate(weights).astype(np.float32),
    )


def skin_samples(model, args, spec=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`cell_samples`, plus which of the two pigment cells each blob is.

    Kept separate rather than widening `cell_samples`: that one is shared by
    several pieces and returns three values everywhere it is
    called. Weight comes from the model here, not a constant, because a skin at
    one areal density has nothing for the log-density map to grade.
    """
    spec = spec or {}
    points, shade = model.cells()
    species = model.species()
    radius = spec.get("cell_radius", args.cell_radius)
    count, per_cell = len(points), spec.get("cell_samples", args.cell_samples)
    index = (np.arange(count, dtype=np.int64)[:, None] * per_cell
             + np.arange(per_cell, dtype=np.int64)[None, :]) % len(DISC)
    samples = (points[:, None, :] + DISC[index] * radius).reshape(-1, 2)
    weights = (np.repeat(model.weights(), per_cell) / per_cell).astype(np.float32)
    return samples, np.repeat(shade, per_cell), weights, np.repeat(species, per_cell)


def tree_samples(
    start: np.ndarray, end: np.ndarray, target: float = 0.6, limit: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Sample every vein segment along its own length.

    By length, not a fixed count per segment: a fixed count turns the long
    segments into dotted rules across the frame, and it reads as a layout bug
    rather than a sampling one. It cost real time once.

    The ceiling is the same trap one level up. A vein or a comet tail is a few
    dozen pixels and never reaches 32 samples, but a microtubule crosses the
    frame: capped at 32 it is drawn as one dot every 56 px, which is the same
    dotted rule and cost an early cover 40,404 samples where the
    honest count is 380,000. An edition that draws long segments says so.
    """
    delta = end - start
    length = np.linalg.norm(delta, axis=1)
    counts = np.clip(np.ceil(length / target), 1, limit).astype(np.int64)
    total = int(counts.sum())
    segment = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    starts = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    fraction = ((np.arange(total, dtype=np.int64) - starts[segment]) / counts[segment]).astype(np.float32)
    return (start[segment] + fraction[:, None] * delta[segment]).astype(np.float32), segment


def with_heads(state, samples, shade, weights, spec):
    """Add the moving object at the front of each trail.

    A blob rather than a point, and heavier than any tail sample, so the
    thing travelling is the brightest thing in the frame and the trail reads
    as belonging to it.
    """
    heads = state.heads()
    radius, per_head = spec["head"], spec.get("head_samples", 26)
    offsets = DISC[:per_head] * radius
    blobs = (heads[:, None, :] + offsets[None, :, :]).reshape(-1, 2).astype(np.float32)
    return (
        np.concatenate((samples, blobs)),
        np.concatenate((shade, np.ones(len(blobs), dtype=np.float32))),
        np.concatenate((weights, np.full(len(blobs), spec.get("head_weight", 2.4), dtype=np.float32))),
    )


def build(name: str, args: argparse.Namespace):
    """Create a fresh model for an edition, deterministically."""
    spec = EDITIONS[name]
    if spec["kind"] == "placozoa":
        return morphogens.Tear(args.height, args.width, seed=args.seed,
                               trail_stride=spec["steps_per_frame"], **spec.get("model", {}))
    if spec["kind"] == "skin":
        # The disc has to arrive at the dish radius on the last frame and not
        # before, so the growth increment is solved from the clip length rather
        # than set: asking for a longer cut grows the same skin more slowly.
        # Linear in radius, because the stripe count goes as the radius.
        #
        # The settle steps grow the disc too, so they belong in the divisor.
        # Leaving them out overshoots: the skin hits the dish radius at frame
        # 207 of 240 and the last second is a disc that has stopped growing.
        frames = round(args.duration * args.fps)
        steps = spec["settle"] + spec["steps_per_frame"] * (frames - 1)
        limit = args.skin_radius
        return morphogens.Stripe(
            args.height, args.width,
            radius=limit,
            seed_radius=args.skin_seed,
            growth=(limit - args.skin_seed) / steps,
        )
    if spec["kind"] == "spiral":
        # One organ per plastochrone, five per frame, so the clip is the whole
        # ontogeny of the head rather than a window onto part of it. The head
        # is exactly full on the last step, which is what fixes the spacing.
        frames = round(args.duration * args.fps)
        return morphogens.Phyllotaxis(
            args.height, args.width, radius=args.head_radius,
            primordia=spec["steps_per_frame"] * (frames - 1) + 1,
        )
    raise ValueError(f"No model for kind {spec['kind']!r}.")


def render_edition(name: str, args: argparse.Namespace) -> Path:
    spec = EDITIONS[name]
    # A `sharp` piece carries its own bloom settings so the cut is reproducible
    # from the edition alone. The cautionary case ships sharp,
    # its filename does not say so, and re-rendering it from the name silently
    # changes the cut.
    bloom_threshold = spec.get("bloom_threshold", args.bloom_threshold)
    bloom_strength = spec.get("bloom_strength", args.bloom_strength)
    width, height = args.width, args.height
    frames = round(args.duration * args.fps)
    caption = build_overlay(width, height, spec, args)
    palette = glow.build_palette(PALETTES[args.palette] if args.palette else spec["palette"])
    palette_b = glow.build_palette(spec["palette_b"]) if "palette_b" in spec else None

    # First pass: run the process to completion. The finished form is both the
    # cover frame and what the exposure is calibrated on, so that the clip
    # brightens into its final state instead of being levelled frame by frame.
    print(f"  {name}: settling", flush=True)
    model = build(name, args)
    # One other piece in the account does this too: the better picture is
    # at the *start*. The hole is what the piece is of, and by the last frame
    # it is nine tenths gone -- a thin band on black, which is a bad thumbnail
    # and a worse first half-second. So the cover and the exposure reference
    # are both read off the opening state and the clip plays away from it.
    if spec.get("cover") == "first":
        model.step(spec["settle"])
    else:
        model.step(spec["settle"] + spec["steps_per_frame"] * (frames - 1))

    if spec["kind"] == "curve":
        final_points = model.points.copy()
        if spec.get("wall"):
            # Confined: the wall fixes the magnification, not the frame. Fitting
            # the bounding box instead would let the dish breathe by a pixel or
            # two per frame as the rim fills in, and the one thing a dish must
            # do is sit still.
            centre = np.array(
                [width * 0.5, height * 0.5], dtype=np.float32
            )
            scale = spec["dish"] * min(width, height) / spec["wall"]
        else:
            span = final_points.max(axis=0) - final_points.min(axis=0)
            centre = (final_points.max(axis=0) + final_points.min(axis=0)) * 0.5
            scale = min(args.fill * width / span[0], args.fill * height / span[1])
        radius = np.linalg.norm(final_points - centre, axis=1).max() * scale
        print(
            f"  {name}: {len(final_points):,} nodes, {scale:.2f} px/unit, "
            f"outermost ink at {radius:.0f} px",
            flush=True,
        )

        def draw(points: np.ndarray, age: np.ndarray) -> np.ndarray:
            samples, segment = curve_samples((points - centre) * scale + np.array([width * 0.5, height * 0.5]))
            # Ranked age, so the palette spreads evenly over the growth history
            # instead of bunching wherever the node count happened to explode.
            rank = (np.argsort(np.argsort(age)) / max(len(age) - 1, 1)).astype(np.float32)
            shade = rank[segment]
            colours = glow.sample_palette(palette, shade)
            colour_sum, density = glow.splat(
                width, height, samples, colours, np.ones(len(samples), dtype=np.float32)
            )
            linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
            linear = glow.bloom(linear, threshold=bloom_threshold, strength=bloom_strength)
            return glow.compose(glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"])), caption)

        samples, _ = curve_samples((final_points - centre) * scale + np.array([width * 0.5, height * 0.5]))
        _, probe = glow.splat(
            width, height, samples, np.zeros((len(samples), 3), dtype=np.float32),
            np.ones(len(samples), dtype=np.float32),
        )
        reference = float(np.percentile(probe[probe > 0], 92.0))
        cover = draw(final_points, model.age)
    elif spec["kind"] in ("vein", "comet", "spindle", "hydra"):

        limit = spec.get("sample_limit", 32)

        def draw_veins(state) -> np.ndarray:
            start, end, shade = state.segments()
            samples, segment = tree_samples(start, end, limit=limit)
            carried = shade[segment]
            taper = spec.get("taper", 0.0)
            weights = (carried ** taper if taper else np.ones(len(samples))).astype(np.float32)
            if spec.get("head"):
                samples, carried, weights = with_heads(state, samples, carried, weights, spec)
            colours = glow.sample_palette(palette, carried)
            colour_sum, density = glow.splat(width, height, samples, colours, weights)
            linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
            linear = glow.bloom(linear, threshold=bloom_threshold, strength=bloom_strength)
            return glow.compose(glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"])), caption)

        start, end, shade = model.segments()
        samples, segment = tree_samples(start, end, limit=limit)
        taper = spec.get("taper", 0.0)
        probe_weights = (shade[segment] ** taper if taper else np.ones(len(samples))).astype(np.float32)
        if spec.get("head"):
            samples, _, probe_weights = with_heads(model, samples, shade[segment], probe_weights, spec)
        _, probe = glow.splat(
            width, height, samples, np.zeros((len(samples), 3), dtype=np.float32), probe_weights
        )
        reference = float(np.percentile(probe[probe > 0], 92.0))
        if spec["kind"] == "comet":
            print(f"  {name}: {model.count:,} bacteria, {len(samples):,} samples", flush=True)
        elif spec["kind"] == "hydra":
            print(f"  {name}: {model.polyps()} polyps, {model.buds()} buds on them, "
                  f"{model.made} buds started, {model.freed} let go, {len(samples):,} samples", flush=True)
        elif spec["kind"] == "spindle":
            print(f"  {name}: {model.bioriented()}/{model.chromosomes} bi-oriented, "
                  f"{model.attached()}/{2 * model.chromosomes} kinetochores held, "
                  f"{len(samples):,} samples", flush=True)
        else:
            print(f"  {name}: {model.count:,} tips, {len(model.sources):,} sources left", flush=True)
        cover = draw_veins(model)
    elif spec["kind"] in ("swarm", "polonaise", "placozoa"):
        # Which state the tone map is calibrated on is a separate question from
        # which state the cover shows. `tear2` opens on five animals with black
        # between them -- the better thumbnail, and the only frame where the
        # text sits on black -- but calibrating on that sparse state blows the
        # clip out as it fills, so the gauge is run to the end regardless.
        gauge = model
        if spec.get("reference") == "last":
            gauge = build(name, args)
            gauge.step(spec["settle"] + spec["steps_per_frame"] * (frames - 1))
        # Two references, both read off the finished plate. Density calibrates
        # the tone map the way it does everywhere else; the colour reference is
        # the extra one this kind needs, because hue carries a quantity that
        # grows over the clip and a per-frame normalisation would flatten the
        # arc it is there to show.
        _, final_travel, _ = gauge.swarm()
        colour_reference = float(np.percentile(final_travel, spec.get("colour_reference", 98.0)))

        def draw_swarm(state) -> np.ndarray:
            samples, shade, weights = swarm_samples(state, spec, colour_reference)
            colours = glow.sample_palette(palette, shade)
            colour_sum, density = glow.splat(width, height, samples, colours, weights)
            linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
            linear = glow.bloom(linear, threshold=bloom_threshold, strength=bloom_strength)
            return glow.compose(glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"])), caption)

        samples, _, weights = swarm_samples(gauge, spec, colour_reference)
        _, probe = glow.splat(
            width, height, samples, np.zeros((len(samples), 3), dtype=np.float32), weights
        )
        reference = float(np.percentile(probe[probe > 0], args.cell_reference))
        grid = model.density()
        print(f"  {name}: {model.count:,} cells, {len(samples):,} samples, "
              f"peak/mean density {np.percentile(grid, 99.9) / grid.mean():.2f}, "
              f"travel reference {colour_reference:.1f} px", flush=True)
        cover = draw_swarm(model)
    elif spec["kind"] == "skin":

        def draw_skin(state) -> np.ndarray:
            samples, shade, weights, species = skin_samples(state, args, spec)
            # One splat, two palettes. Summing the two channels separately --
            # the two-population answer -- is for populations that overlap; two
            # pigment cells never occupy the same patch of skin, so the choice
            # is per cell and a single additive pass keeps a stripe border a
            # border rather than a seam where one layer paints over the other.
            colours = np.where(
                (species == 1)[:, None],
                glow.sample_palette(palette_b, shade),
                glow.sample_palette(palette, shade),
            )
            colour_sum, density = glow.splat(width, height, samples, colours, weights)
            linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
            linear = glow.bloom(linear, threshold=args.bloom_threshold, strength=args.bloom_strength)
            return glow.compose(glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"])), caption)

        samples, _, weights, _ = skin_samples(model, args, spec)
        _, probe = glow.splat(
            width, height, samples, np.zeros((len(samples), 3), dtype=np.float32), weights
        )
        reference = float(np.percentile(probe[probe > 0], args.cell_reference))
        warm = int(model.species().sum())
        print(f"  {name}: {model.count:,} cells, {warm:,} xanthophore, "
              f"radius {model.radius:.0f} px, stripe border {model.boundary():,} px", flush=True)
        cover = draw_skin(model)
    elif spec["kind"] == "sheet":
        # Colour is a rate, not an accumulation, so it is mapped against a
        # fixed reference rather than ranked per frame. Ranking would drag the
        # resting epidermis up the ramp -- it is 84% of the cells and it never
        # contracts -- and a per-frame normalisation would flatten the beat
        # into a constant shimmer. The reference is read off the opening state,
        # which is where the most cells are pulling at once.
        _, opening_rate = model.cells()
        live = opening_rate[opening_rate > 0.0]
        colour_reference = float(np.percentile(live, 98.0)) if len(live) else 1.0

        def draw_sheet(state) -> np.ndarray:
            samples, rate, weights = sheet_samples(state, args, spec)
            shade = np.clip(rate / max(colour_reference, 1e-6), 0.0, 1.0)
            colours = glow.sample_palette(palette, shade)
            colour_sum, density = glow.splat(width, height, samples, colours, weights)
            linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
            linear = glow.bloom(linear, threshold=bloom_threshold, strength=bloom_strength)
            return glow.compose(glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"])), caption)

        samples, _, weights = sheet_samples(model, args, spec)
        _, probe = glow.splat(
            width, height, samples, np.zeros((len(samples), 3), dtype=np.float32), weights
        )
        reference = float(np.percentile(probe[probe > 0], args.cell_reference))
        amnio = int((model.kind == 1).sum())
        print(f"  {name}: {model.count:,} cells, {amnio:,} amnioserosa, "
              f"hole {model.hole() / (width * height) * 100:.1f}% of frame, "
              f"{len(samples):,} samples, colour reference {colour_reference:.3f} px/step",
              flush=True)
        cover = draw_sheet(model)
    elif spec["kind"] in ("cells", "spiral"):

        def draw_cells(state) -> np.ndarray:
            samples, shade, weights = cell_samples(state, args, spec)
            colours = glow.sample_palette(palette, shade)
            colour_sum, density = glow.splat(width, height, samples, colours, weights)
            linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
            linear = glow.bloom(linear, threshold=bloom_threshold, strength=bloom_strength)
            return glow.compose(glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"])), caption)

        samples, _, weights = cell_samples(model, args, spec)
        _, probe = glow.splat(
            width, height, samples, np.zeros((len(samples), 3), dtype=np.float32), weights
        )
        reference = float(np.percentile(probe[probe > 0], args.cell_reference))
        if hasattr(model, "closed"):
            print(f"  {name}: {model.count:,} cells, {len(model.closed)} segments formed", flush=True)
        else:
            print(f"  {name}: {model.count:,} organs, outer radius {model.radii()[0]:.0f} px", flush=True)
        cover = draw_cells(model)
    else:
        channels = field_channels(model, spec, palette, palette_b, height, width, None)
        stacked = sum(channel[0] for channel in channels)
        reference = float(np.percentile(stacked[stacked > 0], 99.0))
        channels = field_channels(model, spec, palette, palette_b, height, width, reference)
        cover = compose_field(channels, reference, spec, args, caption)

    # The slug ends in the palette name, so a --palette cut has to rename that
    # segment or two ramps of one piece write to the same file.
    slug = spec["slug"]
    if args.palette:
        slug = slug.rsplit("_", 1)[0] + "_" + args.palette
    stem = f"{slug}_{width}x{height}_{args.duration:g}s_{args.fps}fps"
    if spec.get("look"):
        # Name the look in the filename. One early piece does not, and re-rendering
        # it from the filename alone changes the cut.
        stem += f"_{spec['look']}"
    if args.hook and spec.get("hook"):
        stem += "_hook_plex"
    if args.tag:
        # A variant cut written alongside the original rather than over it.
        stem += f"_{args.tag}"
    # The cover is the finished form and costs one pass, so it is worth having
    # on its own while tuning: the clip behind it costs two hundred more.
    if args.preview:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        still = args.output_dir / f"{stem}.cover.png"
        Image.fromarray(cover).save(still)
        print(f"Saved {still}")
        return still

    output = args.output_dir / f"{stem}.mp4"
    encoder = start_encoder(output, width, height, args.fps)
    assert encoder.stdin is not None
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cover).save(args.output_dir / f"{stem}.cover.png")
        encoder.stdin.write(cover.tobytes())

        # Second pass, from the same seed, emitting a frame as it goes.
        model = build(name, args)
        if spec["settle"]:
            model.step(spec["settle"])
        for index in range(1, frames):
            if index > 1:
                model.step(spec["steps_per_frame"])
            if spec["kind"] == "curve":
                frame = draw(model.points, model.age)
            elif spec["kind"] == "skin":
                frame = draw_skin(model)
            elif spec["kind"] in ("vein", "comet", "spindle", "hydra"):
                frame = draw_veins(model)
            elif spec["kind"] in ("swarm", "polonaise", "placozoa"):
                frame = draw_swarm(model)
            elif spec["kind"] == "sheet":
                frame = draw_sheet(model)
            elif spec["kind"] in ("cells", "spiral"):
                frame = draw_cells(model)
            else:
                frame = compose_field(
                    field_channels(model, spec, palette, palette_b, height, width, reference),
                    reference, spec, args, caption,
                )
            encoder.stdin.write(frame.tobytes())
            if index % 60 == 0:
                print(f"  {name}: frame {index}/{frames}", flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while rendering {name}.")
    return output


# Named ramps a cut may be rendered against without editing its edition. Use
# with --tag so the variant lands beside the original rather than over it.
PALETTES = {"apex": APEX, "meristem": MERISTEM,
            "kin": KIN, "prism": PRISM, "uv": UV, "fault": FAULT}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=sorted(EDITIONS), action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview", action="store_true", help="Save the cover still and stop.")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--head-radius", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--palette", choices=sorted(PALETTES))
    parser.add_argument("--skin-radius", type=float, default=475.0)
    parser.add_argument("--skin-seed", type=float, default=220.0)
    parser.add_argument("--bloom-threshold", type=float, default=0.30)
    parser.add_argument("--bloom-strength", type=float, default=0.60)
    parser.add_argument("--tag", help="suffix for a variant cut")
    # The house layout, same numbers as the substrate and alife sets. The older
    # wetware cuts predate it and used a symmetric 64 px inset with no scrim
    # and no hook, which is why they need re-rendering rather than patching.
    parser.add_argument("--margin", type=int, default=64)
    parser.add_argument("--title-top", type=int, default=240)
    parser.add_argument("--caption-bottom", type=int, default=190)
    parser.add_argument("--caption-size", type=int, default=27)
    parser.add_argument("--scrim", type=float, default=0.95)
    parser.add_argument("--no-hook", dest="hook", action="store_false")
    parser.add_argument("--hook-size", type=int, default=34)
    parser.add_argument("--hook-gap", type=int, default=82)
    parser.add_argument("--tail-start", type=float, default=520.0)
    parser.add_argument("--tail-end", type=float, default=1400.0)
    parser.add_argument("--densify", type=float, default=1.60)
    parser.add_argument("--cell-samples", type=int, default=10)
    parser.add_argument("--cell-radius", type=float, default=3.1)
    parser.add_argument("--cell-reference", type=float, default=92.0)
    parser.add_argument("--band-top", type=float, default=330.0)
    parser.add_argument("--band-bottom", type=float, default=1400.0)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    for edition_name in options.edition or list(EDITIONS):
        print(f"Rendering {edition_name} ...", flush=True)
        print(f"Saved {render_edition(edition_name, options)}", flush=True)
