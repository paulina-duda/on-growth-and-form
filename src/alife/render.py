#!/usr/bin/env python3
"""Artificial life in the luminous house style: rules that were found, not written.

The substrate and wetware sets film processes biology already knows about. This
one films the other direction -- rules invented in a computer that turn out to
behave like living things, which is the claim artificial life has been making
since Langton and is the only claim in this account that is genuinely arguable.

**The frame carries a population, not an organism.** The other editions compose
a single thing in the middle of a black field; these wrap at the edges, so there
is no middle and no boundary for a structure to have been built against.
`soliton` wraps in both directions. A piece that wraps in one direction only is
held top to bottom by a soft spring, which keeps the black the title and the
hook are set on -- a swarm that fills the frame corner to corner leaves the
typography nothing to sit on.

**Eight seconds, and the clip is a loop**, as everywhere else: it opens on the
organised frame, cuts to the gas it started as, and assembles back into exactly
the frame it opened on.

**This one is paced by the clock, and that is the exception.** Everywhere else
in this project the frames are placed at equal intervals of a measured
progress, because the process accelerates or stalls. Here the population
condenses out of the gas in the first second and then simply swims, at a speed
the friction fixes, so equal steps of the clock already are equal steps of the
process -- and a progress schedule built on how much structure there is would
spend seven of the eight seconds sitting still on the flat part of that curve,
which is to say on the part where all the swimming happens.

**It ends at the top of the population, not at the end of the run.** Left
running, the swimmers meet, merge and thin out, and the frame slowly empties --
true, and a worse picture. The clip stops where there are most of them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import glow
import lenia


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "out"

# One colour per species, and that is the whole colour scheme: there is nothing
# else about a particle to measure. Four hues that stay in the family -- the
# magenta and mint of the wetware set, the amber of the mycelium, and a
# near-white for whichever species ends up as skin.
SPECIES_COLOURS = [
    (255, 62, 168),
    (255, 168, 44),
    (86, 255, 176),
    (214, 240, 255),
]

# The same four species in saturated neon. The soft cut could afford a pale
# near-white species and a muted amber, because the bloom was doing the work and
# every core washed towards white anyway; with the halo turned down those two
# read as pastel and the frame loses its edge. These sit at full chroma so the
# colour survives a tone curve that no longer bleaches it.
NEON_SPECIES = {
    "family": SPECIES_COLOURS,
    "neon": [(255, 0, 140), (0, 245, 255), (170, 255, 0), (150, 40, 255)],
    "signs": [(255, 0, 128), (0, 255, 240), (255, 230, 0), (180, 0, 255)],
    "ember": [(255, 0, 110), (0, 255, 200), (255, 140, 0), (140, 60, 255)],
}

# Found by `swarm.search` with the defaults below -- best of twenty tables drawn
# at random, scored on how much of the population assembles and how fast the
# assemblies then move -- and kept so the piece renders the same thing twice.
# Re-ranked once the frame became a band rather than a torus, because a table
# is only best in the world it was scored in: the same twenty were run again in
# the new geometry at the same density, and this one won there too (1.016; it
# scored 1.459 on the torus, where the coverage term had a bigger frame to fill).
# Run with --search to look for another one.
FOUND_MATRIX = np.array(
    [
        [0.328, -0.189, -0.411, 0.656],
        [-0.450, 0.542, -0.505, -0.958],
        [0.621, -0.755, 0.269, -0.574],
        [-0.388, 0.170, -0.384, 0.418],
    ],
    dtype=np.float32,
)

# The creature this piece is built out of, and everything needed to rebuild it:
# a growth curve, a two-ring kernel, and the six numbers of the arc it starts
# from. Found by `lenia.audition` -- 105 survivors out of 9,000 seeds tried at
# radius 18, of which 98 were still self-limiting when re-run at the radius the
# piece is filmed at, and this one travelled furthest.
FOUND_SOLITON = {
    "mu": 0.2220,
    "sigma": 0.0286,
    "beta": (1.0, 0.66),
    "ring": 0.4655,
    "thickness": 0.2940,
    "lobe": 1.9034,
    "amplitude": 0.8705,
}

# Signed growth, from the wake to the front: near-black where the field is
# dying back, deep magenta through the body that is holding steady, and amber
# into white along the edge the creature is advancing into. The dark half of
# the ramp is long on purpose -- most of a settled colony is growing at very
# nearly zero, and a palette that puts that in the middle turns the whole frame
# into one flat mid-tone.
PLASMA = [(2, 1, 9), (26, 3, 44), (70, 8, 86), (150, 20, 120), (240, 80, 90), (255, 235, 205)]

# Generation: how many builds deep a swimmer is. Crimson for the founders,
# through pink and amber, to cream for the newest. Dust is not a generation at
# all and gets its own stop below the ramp, a dim neutral grey: it is material,
# and a heap of it is as heavy as a swimmer, so the one thing that changes on
# the frame a heap starts to swim is that it takes a colour. The first cut had
# the dust a cool slate, and through the tone curve the unswept sea read as a
# field of blue dots -- blue as the whole of a region, not an accent. Ranked,
# because the counts are exponential: most swimmers at the end are in the last
# two generations, and scaled they would all come out cream.
SWEEP_DUST = (25, 24, 27)

EDITIONS: dict[str, dict] = {
    "soliton": {
        "title": "Soliton",
        "slug": "soliton_lenia_alife",
        "palette": PLASMA,
        "exposure": 0.95,
        "boost": 1.06,
        "caption": (
            "Lenia · continuous automaton (Chan 2019)",
            "one ring kernel · one growth curve",
            "twelve creatures, found by search",
        ),
        "hook": ("Every one of these was stable on its own.",),
    },
}


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def find_encoder() -> tuple[str, list[str]]:
    """Pick an ffmpeg that can actually encode H.264 (conda builds cannot)."""
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
# Shared pieces — the house layout, identical to the substrate set
# --------------------------------------------------------------------------


def build_overlay(width: int, height: int, spec: dict, args) -> Image.Image:
    """Title, hook and data block: Plex, 240 px down, 190 px up, hook centred."""
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


def stamp(radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Offsets and weights that turn a point into a small soft body.

    A particle is a point and a point is one pixel, which the bloom then turns
    into a faint smudge with no core. Splatting a handful of offsets instead
    gives it a body with a bright middle and a soft edge -- and because the
    offsets are fractional, the body does not snap to the pixel grid as the
    thing moves.
    """
    offsets = [(0.0, 0.0)]
    for ring, count in ((radius * 0.6, 6), (radius, 8)):
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        offsets += [(ring * np.cos(a), ring * np.sin(a)) for a in angles]
    weights = np.array(
        [1.0] + [0.55] * 6 + [0.22] * 8, dtype=np.float32
    )
    return np.asarray(offsets, dtype=np.float32), weights / weights.sum() * 2.4


def disc(radius: float, spacing: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """A filled disc of samples, for a thing that has a size rather than a position.

    `stamp` is two rings round a centre, and under the default bloom that
    reads as a soft body. Under `sharp`, at a radius past two pixels, the rings
    stop merging and every point comes out a small asterisk. A grid of samples
    a little under a pixel apart is a disc at any radius, with the weight
    easing off towards the rim so its edge is not cut with scissors.
    """
    span = np.arange(-radius, radius + 1e-9, spacing)
    x, y = np.meshgrid(span, span)
    inside = x**2 + y**2 <= radius**2
    offsets = np.column_stack((x[inside], y[inside])).astype(np.float32)
    weights = 1.0 - 0.5 * (np.hypot(offsets[:, 0], offsets[:, 1]) / radius) ** 2
    return offsets, (weights / weights.sum() * 2.4).astype(np.float32)


def tone(colour_sum: np.ndarray, density: np.ndarray, reference: float, spec: dict, args) -> np.ndarray:
    linear = glow.flame_map(colour_sum, density, reference, boost=spec["boost"])
    threshold = args.bloom_threshold
    strength = args.bloom_strength
    linear = glow.bloom(
        linear,
        threshold=spec.get("bloom_threshold", 0.30) if threshold is None else threshold,
        strength=spec.get("bloom_strength", 0.60) if strength is None else strength,
    )
    return glow.to_bytes(glow.tone_map(linear, exposure=spec["exposure"]))
def soliton_timeline(spec: dict, args) -> tuple[Callable[[float], np.ndarray], np.ndarray]:
    height, width = args.height, args.width
    recipe = FOUND_SOLITON
    world = lenia.Lenia(
        height // 2,
        width // 2,
        radius=args.lenia_radius,
        mu=recipe["mu"],
        sigma=recipe["sigma"],
        beta=recipe["beta"],
    )

    # Twelve copies of one animal, each turned a different way -- `phase` is
    # which way the arc faces and therefore which way the creature swims -- and
    # dealt out with a minimum separation. Left to chance, two of them start
    # close enough to touch in the first half second, and the piece needs them
    # apart long enough to be seen swimming before the first contact.
    generator = np.random.default_rng(args.lenia_seed)
    placed: list[tuple[float, float]] = []
    for _ in range(args.creatures):
        # The requested gap is a wish, not a constraint: twelve discs 170 cells
        # across do not fit in this field however long you shuffle them. Relax
        # it until one lands, rather than quietly dropping the creature -- or,
        # worse, stacking it on top of the last one that did fit.
        separation = args.separation
        while True:
            row = float(generator.integers(0, world.height))
            column = float(generator.integers(0, world.width))
            gaps = [
                np.hypot(
                    min(abs(row - other[0]), world.height - abs(row - other[0])),
                    min(abs(column - other[1]), world.width - abs(column - other[1])),
                )
                for other in placed
            ]
            if not gaps or min(gaps) > separation:
                placed.append((row, column))
                break
            separation *= 0.995
        patch = lenia.crescent(
            args.lenia_radius,
            recipe["ring"],
            recipe["thickness"],
            recipe["lobe"],
            float(generator.uniform(0.0, 2.0 * np.pi)),
            recipe["amplitude"],
        )
        world.place(patch, int(placed[-1][0]), int(placed[-1][1]))

    # The run is a fixed number of steps, and the clip length only decides how
    # fast it is played. Where this piece stops is a composition decision -- far
    # enough for the colonies to be the subject, not so far that they cover the
    # frame and it goes flat -- and tying it to the frame count instead would
    # mean a longer cut quietly becomes a different, worse picture.
    states: list[tuple[np.ndarray, np.ndarray]] = []
    rate = args.lenia_total / args.duration_frames
    advanced = 0
    for frame in range(args.duration_frames):
        target = int(round(rate * (frame + 1)))
        world.step(target - advanced)
        advanced = target
        states.append(
            (
                (np.clip(world.field, 0.0, 1.0) * 255.0).astype(np.uint8),
                # Amplified before it is stored: growth is ±1 at a creature's
                # rim and a few thousandths through the body of a settled
                # colony, so the honest signed value leaves everything but the
                # edges sitting on one palette entry.
                (np.clip(0.5 + 0.5 * args.growth_gain * world.growth, 0.0, 1.0) * 255.0).astype(
                    np.uint8
                ),
            )
        )
    print(
        f"  soliton: {len(placed)} creatures, "
        f"{advanced:,} steps at {rate:.2f} per frame, "
        f"{world.mass() / (world.height * world.width):.0%} of the field alive at the end",
        flush=True,
    )

    palette = glow.build_palette(spec["palette"])
    caption = build_overlay(width, height, spec, args)
    reference = 1.0

    def fields_at(index: int) -> tuple[np.ndarray, np.ndarray]:
        field, growth = states[index]
        return (
            np.repeat(np.repeat(field.astype(np.float32) / 255.0, 2, axis=0), 2, axis=1)[:height, :width],
            np.repeat(np.repeat(growth.astype(np.float32) / 255.0, 2, axis=0), 2, axis=1)[:height, :width],
        )

    def draw(u: float) -> np.ndarray:
        density, shade = fields_at(min(int(u * (len(states) - 1)), len(states) - 1))
        colour_sum = glow.sample_palette(palette, shade) * density[:, :, None]
        return glow.compose(tone(colour_sum, density, reference, spec, args), caption)

    final_density, _ = fields_at(len(states) - 1)
    reference = float(np.percentile(final_density[final_density > 0], 92.0))
    return draw, draw(1.0)
def _centre_of_mass(field: np.ndarray) -> np.ndarray:
    """Where the creature is, on a torus, by the circular mean of each axis.

    An animal sitting on the seam has half its mass at row 2 and half at row
    126, and an arithmetic mean puts its centre in the middle of a world where
    there is nothing at all.
    """
    total = max(float(field.sum()), 1e-9)
    centres = []
    for axis, length in ((1, field.shape[0]), (0, field.shape[1])):
        profile = field.sum(axis=axis)
        angle = np.arange(length) * (2.0 * np.pi / length)
        mean = np.arctan2(
            float((profile * np.sin(angle)).sum() / total),
            float((profile * np.cos(angle)).sum() / total),
        )
        centres.append(float(mean % (2.0 * np.pi)) * length / (2.0 * np.pi))
    return np.asarray(centres, dtype=np.float32)
TIMELINES = {
    "soliton": soliton_timeline,
}


def render_edition(name: str, args: argparse.Namespace) -> Path:
    spec = EDITIONS[name]
    draw, finished = TIMELINES[name](spec, args)

    stem = f"{spec['slug']}_{args.width}x{args.height}_{args.duration:g}s_{args.fps}fps"
    if args.hook and spec.get("hook"):
        stem += "_hook_plex"
    if args.tag:
        # A variant cut of the same piece, written alongside rather than over it.
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
            # Ends on u = 1, the frame the clip opened on: no seam in the loop.
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
    parser.add_argument("--tag", help="suffix for a variant cut, e.g. sharp")
    parser.add_argument(
        "--species-palette", choices=sorted(NEON_SPECIES), default="family",
        help="which four hues the species get",
    )
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--duration", type=float, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold", type=int, default=11, help="frames held on the finished form")
    parser.add_argument("--margin", type=int, default=64)
    parser.add_argument("--title-top", type=int, default=240)
    parser.add_argument("--caption-bottom", type=int, default=190)
    parser.add_argument("--caption-size", type=int, default=27)
    parser.add_argument("--scrim", type=float, default=0.95)
    parser.add_argument("--no-hook", dest="hook", action="store_false")
    parser.add_argument("--hook-size", type=int, default=34)
    parser.add_argument("--hook-gap", type=int, default=82)
    # Default None so a piece may pin its own look in the spec and still be
    # overridden from the command line. A lattice piece pins `sharp`: a lattice
    # of 3 px walls is exactly what a wide halo smears into haze.
    parser.add_argument("--bloom-threshold", type=float, default=None)
    parser.add_argument("--bloom-strength", type=float, default=None)
    # Scaled to the band, not to the frame. What these rules do depends on how
    # many neighbours a particle has, so keeping the count while shrinking the
    # area by two fifths would quietly hand the piece a denser world and a
    # different answer.
    parser.add_argument("--count", type=int, default=11667)
    parser.add_argument("--species", type=int, default=4)
    parser.add_argument("--r-min", type=float, default=7.5)
    parser.add_argument("--r-max", type=float, default=36.0)
    parser.add_argument("--strength", type=float, default=0.50)
    parser.add_argument("--speed-limit", type=float, default=3.0)
    parser.add_argument("--body", type=float, default=3.0, help="radius of a particle's splat")
    parser.add_argument("--trail", type=float, nargs="*", default=[0.5, 0.3, 0.16, 0.08])
    parser.add_argument("--trail-stride", type=int, default=1)
    # The band leaves black for the text to sit on: the title's ink ends at 270,
    # the hook's top line starts near 1460, and nothing may swim through either.
    # Held by a spring rather than a wall, so the population thins into the
    # margin instead of stacking against a line -- which is also why the band is
    # set well inside the clearance it has to keep. A cluster straddling the
    # edge drags its own members out however stiff the spring is, and measured
    # over a whole clip the excursion is about 90 px: with these numbers the
    # furthest any particle ever gets is 293 at the top and 1424 at the bottom.
    parser.add_argument("--band-top", type=float, default=385.0)
    parser.add_argument("--band-bottom", type=float, default=1345.0)
    parser.add_argument("--wall", type=float, default=0.25)
    parser.add_argument("--model-seed", type=int, default=20260823)
    parser.add_argument("--search", action="store_true", help="look for a new interaction table")
    parser.add_argument("--matrix-file", type=Path, help="load tables saved by an earlier --search")
    parser.add_argument("--matrix-rank", type=int, default=0, help="which of the ranked tables to use")
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--search-seed", type=int, default=7)
    # 2,016 particles is 0.035 a square radius: thinner and the world never
    # accumulates cells, thicker and it has
    # organised everything it has by the end of the first quarter.
    parser.add_argument("--soup-count", type=int, default=2016)
    # The radius is the whole scale of the picture: a cell comes out about four
    # radii across, so 30 px puts a cell at roughly 120 px on a 1080 px frame.
    parser.add_argument("--soup-radius", type=float, default=30.0)
    parser.add_argument("--soup-total", type=int, default=3000, help="steps in the whole run")
    # The drive. 0.3 a step is the smallest trickle that keeps cells condensing
    # for the whole clip: at 0 the count stalls (13 at the halfway point and 13
    # at the end), at 1.0 the world floods and the cells merge back into mush.
    parser.add_argument("--soup-inject", type=float, default=0.3)
    parser.add_argument("--soup-seeds", type=int, default=3, help="cells present at the start")
    parser.add_argument("--soup-seed", type=int, default=11)
    parser.add_argument("--soup-body", type=float, default=3.0)
    # Twenty-four, measured rather than picked: at 60 the chaotic cores merge
    # into one grey mass well before the clip ends and the roads have nothing
    # to be seen against; at 24 they stay separable for the whole 8 s.
    parser.add_argument("--ants", type=int, default=24)
    # 260 x 229 frames = 59,540 steps. A lone ant needs about 10,000 to leave
    # its mess and start a road, so the first ones launch around frame 40
    # and the rest keep launching for the length of the clip.
    parser.add_argument("--ant-steps", type=int, default=260, help="ant steps per frame")
    parser.add_argument("--ant-seed", type=int, default=3)
    parser.add_argument("--ant-spread", type=float, default=0.30, help="starting band, of the frame")
    # Steps for a trail to fade from white back into the violet of everything
    # already built. About a second of clip: long enough to see the bright tip
    # travelling down a road, short enough that the road behind it is history.
    parser.add_argument("--ant-recency", type=float, default=8000.0)
    parser.add_argument("--lenia-radius", type=float, default=30.0)
    parser.add_argument(
        "--lenia-total", type=int, default=480,
        help="simulation steps in the whole run, spread over however many frames",
    )
    parser.add_argument("--growth-gain", type=float, default=0.8)
    parser.add_argument("--creatures", type=int, default=12)
    parser.add_argument("--separation", type=float, default=170.0, help="minimum gap between seeds")
    parser.add_argument("--lenia-seed", type=int, default=4)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--ga-seed", type=int, default=11)
    parser.add_argument("--gutter", type=int, default=10, help="black between panels")
    parser.add_argument("--grid-top", type=int, default=90, help="gap under the title")
    parser.add_argument("--grid-bottom", type=int, default=320, help="gap over the hook")
    parser.add_argument("--tree-top", type=int, default=90, help="gap under the title")
    parser.add_argument("--tree-bottom", type=int, default=250, help="gap over the data block")
    parser.add_argument("--tree-inset", type=int, default=20)
    parser.add_argument("--spacing", type=float, default=1.5, help="pixels between splatted points")
    parser.add_argument("--ink", type=float, default=0.85)
    parser.add_argument("--thickness", type=float, default=1.1, help="pixels either side of a line")
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--boost", type=float)
    args = parser.parse_args()
    args.duration_frames = round(args.duration * args.fps)
    return args


if __name__ == "__main__":
    options = parse_args()
    for edition_name in options.edition or list(EDITIONS):
        for key in ("exposure", "boost"):
            if getattr(options, key) is not None:
                EDITIONS[edition_name][key] = getattr(options, key)
        print(f"Rendering {edition_name} ...", flush=True)
        print(f"Saved {render_edition(edition_name, options)}", flush=True)
