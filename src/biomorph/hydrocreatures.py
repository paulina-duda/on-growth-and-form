#!/usr/bin/env python3
"""Three hydrocreatures that swim on their own equation, and a loop that closes.

Inspired by the generative sketches of @yuruyurau, who posts them on X at
x.com/yuruyurau. None of his code was used: the curve, the harmonic search, the
loop-closure work, the composition and the rendering here are mine. What is
borrowed is the idea of a single closed curve swept densely and drawn as
strokes. This supersedes an earlier `hydrocreatures` cut -- deleted rather than
kept, because nothing in it survived unchanged -- and it differs from that cut
in the two things the source sketch is actually about:

- **The motion is the equation's, not a path laid over it.** The earlier cut
  subtracted the cloud's median every frame, which deletes the sketch's own
  travel, and then put a designed swim path back on top. Here the skeleton term
  `99*sin(C)`, `99*sin(4C)` is left to carry the animal, exactly as the sketch
  has it: the creature goes where `C` takes it. Nothing schedules a meeting and
  nothing steers.

- **The clip closes.** The sketch's `-t/48` inside `C` has period 96*pi, so no
  reel-length clip ever returns to its first frame: the earlier cut measured
  its own seam at 1.72 radii and recorded that ten seconds bought nothing over
  eight. Two facts fix it without touching the shape of the equation:

    * `sin(4C)` returns exactly when `4*LEAN*T` is a multiple of 2*pi. At
      LEAN = 1/24 and T = 12*pi (240 frames of the source's own pi/20 step)
      that is 2*pi on the nose, so the vertical skeleton -- the term that
      carries nearly all of the travel -- is periodic over the clip by
      construction.
    * `sin(C)` then advances by exactly pi/2, and sin returns across a pi/2
      advance only from one place: `C` must start at 3*pi/4 (mod pi). `C`
      starts at mean(d)/9 + m, so **m is derived, not chosen**: m = 3*pi/4 -
      mean(d)/9 (mod pi), refined against the measured seam.

  What is left over is second order and it does not go away: `C` is d/9 + m and
  `d` is spread across the curve, so the root can only be centred, never hit by
  every point at once. The three animals here close to 0.097, 0.099 and 0.076
  body radii, against 7.99 for the same equation at the sketch's own lean over
  the same eight seconds -- both measured the way `seam` below measures, which
  is stricter than the earlier cut's and not the same number as its 1.72.

  The figure that says whether any of that reads is the seam against an
  ordinary frame: **1.3, 1.5 and 1.9 frame steps**, all inside the clip's own
  p90 step. The wrap is one slightly long frame, not a cut.

  LEAN = 1/24 is twice the sketch's rate and the only change to the equation:
  12.6 body radii of travel per clip rather than 8.3. Nothing else moves.

**The genome is (a, b, c, f) and it was searched, not inherited.** `k` and `e`
are 9*cos(a*i)*sin(b*i) and 9*cos(c*i)*sin(f*i), so the curve is 2*pi-periodic
in `i` for any integer harmonics -- the sketch's (5,1,4,3) is one member of a
family, and not a good one here. All 1,260 sets with harmonics up to 6 were
screened at the two m that close the loop; 2,482 of the 2,520 (set, branch)
pairs admit one. They were then scored on the **worst** moment of the clip rather
than the average, because an animal that is a bell for half its cycle and a
cross for the other half passes on averages and fails on screen -- which is
what the first screen did, and what the contact sheet caught:

    harmonics    branch   upright   symmetry   compactness   aspect      swing
    (2,2,3,6)      0       0.98       0.82        0.26      0.30-0.67    2.7x
    (6,6,1,2)      0       0.98       0.90        0.25      0.30-0.67    2.7x
    (2,6,4,1)      1       0.91       0.88        0.22      0.31-0.95    4.4x
    (5,1,4,3)      1       0.82       0.85        0.34      0.22-0.90    4.0x

Compactness is mean radius over the 0.985 quantile: 0.5 is a ring, 0.15 is a
spindly cross. **Every genome in this family goes spindly somewhere in its
cycle** -- the exponent `p` sweeps d^+1 to d^-1 every 40 frames -- so the
screen ranks the worst moment rather than forbidding it: the best in the whole
search is 0.50, and the eleven sets that clear the other four gates sit at
0.21 to 0.31. The sketch's own genome is on that line for compactness and
fails the other two: at its closing m the body tips off vertical (0.82) and
narrows to 0.22 of its own width at some phase, which is the spike the earlier
cut's note reports in one of the sketch's own three. These three
were chosen off the contact sheet, from the eleven, for reading as three
different animals. Harmonics that share a common factor trace the same curve
twice, so (4,4,2,4) is (2,2,1,2) and the space is smaller than it looks.

The two m branches sit pi apart and drift in antiphase, so the third animal
rises while the other two fall.

Per animal, `phase` offsets the pulse without moving the skeleton: shifting `t`
by p and m by LEAN*p leaves `C = d/9 - LEAN*t + m` untouched while the pulse
`sin(t/2+m)` and the exponent both move. That is the only way to desynchronise
three animals whose m is pinned by the seam.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/on-growth-and-form-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/on-growth-and-form-cache")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import glow


OUTPUT_DIR = Path(__file__).parents[2] / "out"

TITLE = "Hydrocreatures"
# Shortened from four lines to three: the stretch/lean line (`p = d^sin(d²−t+m)
# C = d/9 − t/24 + m`) dropped entirely rather than just its labels. `p` and
# `C` still feed `travel` below, they just aren't spelled out on screen anymore.
CAPTION = (
    "body     k = 9·cos(ai)·sin(bi)   e = 9·cos(ci)·sin(fi)",
    "breath   d = hypot(k,e)³/999 + 1.2 − sin(t/2+m)³/4",
    "travel   x = 99·sin(C) + k·p   y = 99·sin(4C) + e·p",
)


MARGIN = 64
CAPTION_SIZE = 27
HOOK_GAP = 82

FONT_FAMILIES = {
    "dejavu": (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    ),
    "plex": (
        Path(__file__).parents[1] / "fonts" / "IBMPlexMono-Regular.ttf",
        Path(__file__).parents[1] / "fonts" / "IBMPlexMono-Bold.ttf",
    ),
}

# --- the equation -----------------------------------------------------------

# The source's own step, kept. `t += PI/20` per drawn frame is the motion, not
# a detail: sin(t/2+m) pulses the body three times over the clip, the exponent
# sin(d*d-t+m) runs six cycles, and both close exactly over 240 frames.
DT = np.pi / 20.0

# The one change to the equation, and it is what makes the clip loop. See the
# module docstring: 4*LEAN*(240*DT) = 2*pi exactly.
LEAN = 1.0 / 24.0

N_I = 11000                # samples along the closed curve
SHEETS = 3                 # copies at nearby equation times: rings and filaments
SHEET_SPAN = 0.34
W_SHEET = 0.34

# How much of the pulse to draw. The drawn radius is (e_t / e_max)**PULSE_GAIN,
# so 1.0 is the pulse at full depth and 0.0 is what the sketch does -- dividing
# by the current extent, which cancels the pulse and leaves an animal that
# changes shape without ever changing size. These three swing 2.7-4.4x; at 0.70
# that is drawn as 2.0-3.0x, which reads without emptying the frame.
PULSE_GAIN = 0.70

# The equation's travel is 12.6 body radii per clip and the frame is 3.6 radii
# tall at these display scales, so the animal's own excursion has to be scaled
# to fit the frame it is drawn in -- the sketch's canvas is 400 wide with a
# creature 24 across, this one is 1080 wide with creatures 320 across. TRAVEL
# is that ratio, not a decision about how fast anything swims: the path shape,
# its timing and its closure are the equation's.
#
# 0.35 until the data block went back to four lines, which lifted the hook to
# row 1471 and left ten pixels over the animals -- less than the bloom bleeds.
# The 79 px came off the excursion rather than off the animals, because the
# frame is already only 2.5% lit and the three read small.
TRAVEL = 0.30

# harmonics (a, b, c, f), branch, phase, home in model units, display scale.
# The branch picks which of the two seam roots the animal sits on -- they are pi
# apart and drift in antiphase, so branch 1 rises while branch 0 falls. m itself
# is not stored: `refine_m` derives it at render time, because the root moves
# with `phase` (see that function) and a stored m would quietly stop closing the
# loop the moment anything else about the animal changed.
CREATURES = (
    ((2, 2, 3, 6), 0, 0.0, (-0.48, 0.466), 0.460),
    ((6, 6, 1, 2), 0, 7.4, (0.50, 0.066), 0.460),
    ((2, 6, 4, 1), 1, 3.1, (-0.02, -0.394), 0.400),
)

# One hue per animal, because the only thing separating them is the genome --
# so colour names which equation you are looking at, and the ramp inside each
# hue is the ranked stretch term. Low ends stay dark; only the cores reach
# white. Blue appears once, as one animal's accent, never as the piece: `reef`
# spends that allowance on its third animal, `aurora` does not spend it at all.
#
# Two sets, and the pairing with a hook is deliberate rather than decorative.
# `aurora` separates the three animals as far as the wheel allows -- three
# strangers -- which is the version to run under the hook about the drift being
# a term. `reef` runs them warm-to-cool through one arc, which reads as three of
# a kind and suits the hook about whose intention this is.
VARIANTS = {
    "aurora": dict(
        hook=("The drift is not a decision. It is a term.",),
        palettes=(
            [(2, 14, 16), (8, 66, 74), (18, 148, 148), (54, 216, 198), (170, 246, 232), (240, 255, 252)],
            [(18, 8, 2), (84, 42, 6), (170, 96, 14), (238, 158, 40), (255, 212, 130), (255, 246, 224)],
            [(5, 2, 18), (70, 12, 78), (168, 30, 148), (240, 84, 190), (255, 176, 224), (255, 240, 250)],
        ),
    ),
    "reef": dict(
        hook=("Nothing here intends anything. You do.",),
        palettes=(
            [(2, 16, 6), (10, 74, 28), (26, 158, 62), (86, 230, 120), (190, 252, 206), (244, 255, 246)],
            [(20, 4, 4), (92, 18, 10), (182, 48, 26), (244, 104, 62), (255, 178, 142), (255, 240, 230)],
            [(6, 4, 20), (30, 26, 92), (58, 66, 190), (110, 130, 246), (188, 206, 255), (240, 246, 255)],
        ),
    ),
    # Neon: the same three ramps pushed to the corners of the gamut -- full
    # cyan, full magenta, acid lime -- with the low ends dropped nearly to
    # black. The ranked stretch term is unchanged, so what reads as harder edges
    # is the ramp spending less of itself on the mid tones and more on the jump
    # from dark to core. Only the palette changes; exposure, boost and both
    # bloom knobs are identical across the three cuts, so the comparison is of
    # colour and nothing else.
    # Shipped 2026-09-05 with `reef`'s hook rather than its own -- the pairing
    # picked on review was this palette's colour with that line's sense, not
    # either variant's original coupling.
    "neon": dict(
        hook=("Nothing here intends anything. You do.",),
        palettes=(
            [(0, 8, 12), (0, 54, 82), (0, 148, 202), (0, 228, 255), (150, 250, 255), (240, 255, 255)],
            [(12, 0, 10), (76, 0, 48), (188, 0, 118), (255, 40, 170), (255, 150, 215), (255, 236, 248)],
            [(8, 12, 0), (44, 68, 0), (126, 178, 0), (200, 255, 40), (236, 255, 150), (250, 255, 226)],
        ),
    ),
    # Review-only: all three cornflower blue, to see how one hue across every
    # animal reads against the "hue names the genome" rule above. No hook --
    # a colour test, not a candidate. The three ramps differ slightly in hue
    # lean (violet-blue, canonical cornflower, cyan-blue) so they can still be
    # told apart, but all sit in the one family the account's rule 6 calls
    # an accent, not a piece.
    "cornflower": dict(
        hook=(),
        palettes=(
            [(4, 4, 14), (16, 18, 52), (38, 46, 110), (74, 96, 190), (130, 160, 230), (230, 238, 255)],
            [(5, 5, 16), (20, 22, 58), (46, 58, 124), (100, 149, 237), (170, 200, 250), (235, 242, 255)],
            [(3, 4, 18), (14, 20, 64), (34, 54, 132), (84, 120, 210), (150, 185, 245), (232, 240, 255)],
        ),
    ),
    # Review-only: no hue at all, to see whether the three read apart on shape
    # and brightness alone once colour stops doing the work. Not a candidate --
    # a mono variant with no hook and a slight warm/cool split per animal so
    # the three can still be told apart in a still.
    "mono": dict(
        hook=(),
        palettes=(
            [(4, 5, 6), (30, 34, 38), (80, 88, 96), (150, 158, 166), (215, 220, 224), (250, 251, 252)],
            [(5, 5, 5), (34, 34, 34), (90, 90, 90), (160, 160, 160), (220, 220, 220), (255, 255, 255)],
            [(6, 5, 4), (38, 34, 30), (96, 88, 80), (166, 158, 150), (224, 220, 215), (252, 251, 250)],
        ),
    ),
}


def state(harmonics: tuple[int, int, int, int], m: float, t: float, n: int = N_I):
    """The source procedure with the harmonics opened up, `i` swept as the
    continuous parameter it is, and the skeleton left in."""
    a, b, c, f = harmonics
    i = np.linspace(0.0, 2.0 * np.pi, n)
    k = 9.0 * np.cos(a * i) * np.sin(b * i)
    e = 9.0 * np.cos(c * i) * np.sin(f * i)
    d = np.hypot(k, e) ** 3 / 999.0 + 1.2 - np.sin(t / 2.0 + m) ** 3 / 4.0
    curve = d / 9.0 - t * LEAN + m
    p = d ** np.sin(d * d - t + m)
    # p5.js points its vertical axis down.
    return 99.0 * np.sin(curve) + k * p, -(99.0 * np.sin(curve * 4.0) + e * p), p


def seam_root(harmonics: tuple[int, int, int, int], branch: int = 0) -> float:
    """The m at which sin(C) survives the clip: C must start at 3*pi/4 (mod pi).

    Kept as a function even though `CREATURES` carries the values refined
    against the measured seam, so a fourth animal is derived rather than
    guessed. The two branches sit pi apart and drift in antiphase.
    """
    a, b, c, f = harmonics
    i = np.linspace(0.0, 2.0 * np.pi, 2000)
    k = 9.0 * np.cos(a * i) * np.sin(b * i)
    e = 9.0 * np.cos(c * i) * np.sin(f * i)
    d_mean = float(np.mean(np.hypot(k, e) ** 3 / 999.0 + 1.2))
    return (3.0 * np.pi / 4.0 - d_mean / 9.0) % np.pi + branch * np.pi


def seam(harmonics, m: float, phase: float, frames: int, n: int = 1600) -> float:
    """How far the animal is from where it started, in body radii.

    Point-wise, not a set distance: both clouds are the same `i` grid on the
    same curve, and `d` closes exactly over the clip, so point j at the end is
    the same point of the animal as point j at the start. A set distance would
    forgive the curve sliding along itself; the eye does not.
    """
    m_eff = shifted_m(m, phase)
    first = np.column_stack(state(harmonics, m_eff, animal_clock(0, phase), n)[:2])
    last = np.column_stack(state(harmonics, m_eff, animal_clock(frames, phase), n)[:2])
    extent = np.quantile(np.linalg.norm(first - np.median(first, axis=0), axis=1), 0.985)
    return float(np.linalg.norm(last - first, axis=1).mean() / extent)


def refine_m(harmonics, branch: int, phase: float, frames: int) -> tuple[float, float]:
    """Land m on the seam root, for this animal's phase.

    `seam_root` is exact only for a body of constant `d`. The real `d` carries
    the pulse -- -sin(t/2+m)**3/4, so up to 0.25 of it, which is 0.028 rad once
    divided by 9 -- and the animal's phase decides where in that pulse the clip
    begins. So the analytic root is the starting point and the measured seam
    picks the winner: it moves the root by up to 0.03 of m, which is worth
    about a third of the seam.
    """
    best = (np.inf, seam_root(harmonics, branch))
    for step in (0.004, 0.0005):
        centre = best[1]
        for m in centre + np.arange(-15, 16) * step:
            value = seam(harmonics, float(m), phase, frames)
            if value < best[0]:
                best = (value, float(m))
    return best[1], best[0]


def ranks(values: np.ndarray) -> np.ndarray:
    """Rank the skewed stretch term rather than scaling it -- house rule 2."""
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values))
    out[order] = np.linspace(0.0, 1.0, len(values))
    return out


def strokes(px, py, val, limit=160.0):
    """Resample the polyline so a filament splats as a filament, not as beads.

    Sampled by length: a fixed count per segment turns the long ones into
    dotted rules across the frame, which reads as a layout bug.
    """
    points = np.column_stack((px, py))
    u, v = points[:-1], points[1:]
    length = np.linalg.norm(v - u, axis=1)
    keep = length < limit                      # a chord no part of the animal travelled
    u, v, length = u[keep], v[keep], length[keep]
    va, vb = val[:-1][keep], val[1:][keep]
    counts = np.clip((length / 0.9).astype(int) + 1, 1, 40)
    index = np.repeat(np.arange(len(u)), counts)
    frac = np.concatenate([np.linspace(0, 1, q, endpoint=False) for q in counts])[:, None]
    return (u[index] + (v[index] - u[index]) * frac,
            va[index] + (vb[index] - va[index]) * frac[:, 0])


def animal_clock(index: int, phase: float) -> float:
    return phase + index * DT


def shifted_m(m: float, phase: float) -> float:
    """Move the pulse without moving the skeleton.

    `C` carries -LEAN*t + m, so advancing the animal's clock by `phase` and its
    m by LEAN*phase leaves `C` exactly where it was while the pulse and the
    exponent both shift. Without this the three animals -- whose m is pinned to
    within two values by the seam -- would beat in unison.
    """
    return m + LEAN * phase


def plan_clip(frames: int) -> list[dict]:
    """Everything about the three animals that has to be measured once.

    `m` lands on the seam root for that animal's phase. The scale is the
    animal's largest state over the clip, not each frame's own extent: the
    sketch divides by the current extent, and that is what cancels the pulse.
    The track is the equation's own position -- the median of the cloud -- taken
    relative to frame one, so the animal starts at its designed home and then
    goes wherever `C` takes it.
    """
    plan: list[dict] = []
    for harmonics, branch, phase, home, display in CREATURES:
        m, closure = refine_m(harmonics, branch, phase, frames)
        m_eff = shifted_m(m, phase)
        centres = np.empty((frames, 2))
        extents = np.empty(frames)
        steps = np.empty(frames - 1)
        previous = None
        for index in range(frames):
            x, y, _ = state(harmonics, m_eff, animal_clock(index, phase), 2200)
            cloud = np.column_stack((x, y))
            centre = np.median(cloud, axis=0)
            centres[index] = centre
            extents[index] = np.quantile(np.linalg.norm(cloud - centre, axis=1), 0.985)
            if previous is not None:
                steps[index - 1] = np.linalg.norm(cloud - previous, axis=1).mean()
            previous = cloud
        scale = float(extents.max())
        # The seam only reads as a jump if it is bigger than an ordinary frame,
        # so that -- not the seam in radii -- is the number to judge it by.
        step = float(np.median(steps) / extents[0])
        plan.append(dict(harmonics=harmonics, m=m, phase=phase, home=home, display=display,
                         seam=closure, seam_steps=closure / step, scale=scale,
                         swing=float(extents.max() / extents.min()),
                         track=(centres - centres[0]) / scale))
    return plan


def sample_frame(index: int, half: float, plan: list[dict], palettes, vshift: float = 0.0):
    """One frame, per animal, so each keeps its own hue."""
    S: list[np.ndarray] = []
    C: list[np.ndarray] = []
    W: list[np.ndarray] = []
    for j, animal in enumerate(plan):
        harmonics, phase, display = animal["harmonics"], animal["phase"], animal["display"]
        m_eff = shifted_m(animal["m"], phase)
        t = animal_clock(index, phase)
        x, y, _ = state(harmonics, m_eff, t)
        cloud = np.column_stack((x, y))
        centre = np.median(cloud, axis=0)
        extent_now = float(np.quantile(np.linalg.norm(cloud - centre, axis=1), 0.985))
        # Part of the pulse, not all of it: see PULSE_GAIN.
        divisor = extent_now ** (1.0 - PULSE_GAIN) * animal["scale"] ** PULSE_GAIN
        seat = np.array(animal["home"]) + animal["track"][index] * TRAVEL * display
        for sheet in range(SHEETS):
            offset = (sheet / (SHEETS - 1) - 0.5) if SHEETS > 1 else 0.0
            x, y, p = state(harmonics, m_eff, t + offset * SHEET_SPAN)
            body = (np.column_stack((x, y)) - centre) / divisor * display + seat
            pts, vals = strokes(body[:, 0] * half + half,
                                half * 1920 / 1080 + vshift - body[:, 1] * half,
                                ranks(np.abs(p - 1.0)))
            if not len(pts):
                continue
            S.append(pts)
            C.append(glow.sample_palette(palettes[j], vals))
            W.append(np.full(len(pts), W_SHEET))
    return (np.concatenate(S).astype(np.float32),
            np.concatenate(C).astype(np.float32),
            np.concatenate(W).astype(np.float32))


def render_buffer(index, args, plan, palettes, reference):
    S, C, W = sample_frame(index, args.width / 2, plan, palettes, args.vshift)
    colour_sum, density = glow.splat(args.width, args.height, S, C, W)
    mapped = glow.flame_map(colour_sum, density, reference, boost=args.boost)
    mapped = glow.bloom(mapped, threshold=args.bloom_threshold, strength=args.bloom_strength)
    return glow.to_bytes(glow.tone_map(mapped, exposure=args.exposure)), density


def draw_hook(overlay: Image.Image, args: argparse.Namespace) -> None:
    if not args.hook:
        return
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(str(glow.MONO_FONT), args.hook_size)
    text = "\n".join(args.hook)
    spacing = max(6, args.hook_size // 3)
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    if args.no_caption:
        # No data block to clear, so the hook takes the block's own anchor
        # (bottom margin) rather than sitting HOOK_GAP above an empty one --
        # that left it stranded high up over a dead patch of frame.
        top = overlay.height - args.caption_bottom - box[3] - 4
    else:
        caption_font = ImageFont.truetype(str(args.equation_face), CAPTION_SIZE)
        caption_box = draw.multiline_textbbox(
            (0, 0), "\n".join(CAPTION), font=caption_font, spacing=max(4, CAPTION_SIZE // 3)
        )
        caption_ink_top = overlay.height - args.caption_bottom - caption_box[3] - 4 + caption_box[1]
        top = caption_ink_top - HOOK_GAP - box[3]
    draw.multiline_text(
        (overlay.width / 2 - (box[0] + box[2]) / 2, top),
        text, font=font, fill=(255, 255, 255, 244), spacing=spacing, align="center",
        stroke_width=4, stroke_fill=(0, 0, 0, 165),
    )


def text_layer(args: argparse.Namespace) -> Image.Image:
    """The overlay with the scrim off, so `report` measures ink and not the veil."""
    scrim, args.scrim = args.scrim, 0.0
    try:
        return build_overlay(args)
    finally:
        args.scrim = scrim


def build_overlay(args: argparse.Namespace) -> Image.Image:
    overlay = glow.make_caption(
        args.width, args.height, TITLE, () if args.no_caption else CAPTION,
        equation_size=CAPTION_SIZE, margin=MARGIN, equation_face=args.equation_face,
        top_margin=args.title_top, bottom_margin=args.caption_bottom, scrim=args.scrim,
    )
    draw_hook(overlay, args)
    return overlay


def find_encoder() -> tuple[str, list[str]]:
    """Pick an ffmpeg that can actually encode H.264 -- a conda build once could not."""
    candidates = [
        str(Path(directory) / "ffmpeg")
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if (Path(directory) / "ffmpeg").is_file()
    ]
    candidates.append("/usr/bin/ffmpeg")
    for ffmpeg in dict.fromkeys(candidates):
        try:
            encoders = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        if "libx264" in encoders:
            return ffmpeg, ["-c:v", "libx264", "-crf", "17", "-preset", "slow"]
    raise RuntimeError("No ffmpeg with libx264 found.")


def report(frames: int, args: argparse.Namespace, plan: list[dict], palettes) -> None:
    """Say what the numbers are before spending twenty minutes on the render."""
    half = args.width / 2
    for animal in plan:
        display = animal["display"]
        track = animal["track"] * TRAVEL * display
        span = track.max(axis=0) - track.min(axis=0)
        arc = float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())
        close = float(np.linalg.norm(track[-1] - track[0]))
        print(f"  {animal['harmonics']} m={animal['m']:.4f}: seam {animal['seam']:.4f} radii "
              f"= {animal['seam_steps']:.2f} frame steps, "
              f"scale {animal['scale']:.1f}, pulse {animal['swing']:.1f}x, "
              f"travel {arc / (TRAVEL * display):.1f} radii, drawn span "
              f"({span[0] * half:.0f}, {span[1] * half:.0f}) px, "
              f"seat step at the seam {close * half:.1f} px", flush=True)
    # Where the animals actually reach, against the frame and the text zones.
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for index in range(0, frames, 6):
        S, _, _ = sample_frame(index, half, plan, palettes, args.vshift)
        lo = np.minimum(lo, S.min(axis=0))
        hi = np.maximum(hi, S.max(axis=0))
    print(f"  ink box over the clip: x {lo[0]:.0f}..{hi[0]:.0f} of {args.width}, "
          f"y {lo[1]:.0f}..{hi[1]:.0f} of {args.height}", flush=True)
    # The text zones are measured off the layer itself rather than guessed from
    # the margins: a hook moves with the height of the data block above it, so
    # the estimate that used to be printed here stopped being true the moment
    # this cut grew a hook and lost a caption line.
    text = np.asarray(text_layer(args))[:, :, 3] > 40
    rows = np.where(text.any(axis=1))[0]
    blocks = np.split(rows, np.where(np.diff(rows) > 40)[0] + 1)
    zones = " | ".join(f"{b.min()}..{b.max()}" for b in blocks)
    above = lo[1] - blocks[0].max()
    below = (blocks[1].min() if len(blocks) > 1 else args.height) - hi[1]
    print(f"  text ink rows: {zones}  ->  {above:.0f} px under the title, "
          f"{below:.0f} px over the {'hook' if args.hook else 'data block'} "
          f"({'clear' if min(above, below) > 20 else 'TOO TIGHT'}, bloom bleeds 10-20)",
          flush=True)


def render(args: argparse.Namespace) -> Path:
    frames = round(args.duration * args.fps)
    if frames % 240:
        print(f"  warning: {frames} frames is not 240; the loop closes only at 240 "
              f"({240 * DT / np.pi:.0f}pi of equation time)", flush=True)
    palettes = [glow.build_palette(stops) for stops in args.palettes]
    print("  measuring the clip...", flush=True)
    plan = plan_clip(frames)
    report(frames, args, plan, palettes)

    # One density reference for the whole clip, taken off frame one: a per-frame
    # reference makes the tone curve breathe with the animal and undoes the
    # point of having kept the pulse.
    S, C, W = sample_frame(0, args.width / 2, plan, palettes, args.vshift)
    _, density = glow.splat(args.width, args.height, S, C, W)
    reference = float(np.percentile(density[density > 0], 99.0))
    print(f"  density reference (p99 of frame one): {reference:.2f}", flush=True)

    overlay = build_overlay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def draw(index: int) -> np.ndarray:
        buffer, _ = render_buffer(index, args, plan, palettes, reference)
        return glow.compose(buffer, overlay)

    if args.stills:
        for index in args.stills:
            Image.fromarray(draw(index)).save(
                args.output.with_name(f"{args.output.stem}_f{index:03d}.png"))
            print(f"  still {index}", flush=True)
        return args.output

    cover = draw(0)
    Image.fromarray(cover).save(args.output.with_name(args.output.stem + "_cover.png"))

    ffmpeg, codec = find_encoder()
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
        "-an", *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    try:
        encoder.stdin.write(cover.tobytes())
        for index in range(1, frames):
            encoder.stdin.write(draw(index).tobytes())
            if index % 30 == 0:
                print(f"  frame {index}/{frames}", flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed while rendering the hydrozoa.")
    return args.output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three hydrocreatures on their own equation.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="aurora",
                        help="which hook and palette set to cut; also names the file")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--stills", type=int, nargs="*", default=None,
                        help="render these frame indices as PNGs instead of a clip")
    parser.add_argument("--title-top", type=int, default=240)
    parser.add_argument("--caption-bottom", type=int, default=None,
                        help="default 190 with a data block, 270 without one (see --no-caption)")
    parser.add_argument("--no-caption", action="store_true",
                        help="drop the data block; title and hook only")
    parser.add_argument("--vshift", type=float, default=None,
                        help="push the three animals down this many px; default 0 with a data block, "
                             "68.5 without one (see --no-caption)")
    parser.add_argument("--scrim", type=float, default=0.95)
    parser.add_argument("--hook-size", type=int, default=34)
    parser.add_argument("--font", choices=sorted(FONT_FAMILIES), default="plex")
    parser.add_argument("--equation-font", choices=sorted(FONT_FAMILIES), default="plex")
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--boost", type=float, default=1.05)
    parser.add_argument("--bloom-threshold", type=float, default=0.32)
    parser.add_argument("--bloom-strength", type=float, default=0.55)
    args = parser.parse_args(argv)
    args.hook = VARIANTS[args.variant]["hook"]
    args.palettes = VARIANTS[args.variant]["palettes"]
    if args.caption_bottom is None:
        # 190 is every other piece's data-block margin. Without a data block
        # the hook sits on that same anchor (see draw_hook), and 190 there
        # posted straight into Instagram's own account-name/audio overlay --
        # measured on the live post, not guessed. 270 clears it.
        args.caption_bottom = 270 if args.no_caption else 190
    if args.vshift is None:
        # Moving the hook to 270 shrinks the gap below the animals without
        # touching the gap below the title, so re-centre by the same amount
        # the hook moved: 68.5 puts both gaps at ~123-124 px, measured on
        # the no-caption cut.
        args.vshift = 68.5 if args.no_caption else 0.0
    if args.output is None:
        args.output = OUTPUT_DIR / (
            f"growth-form_hydrozoa-closedloop_hook-{args.variant}"
            f"_{args.width}x{args.height}_{args.duration:g}s_{args.fps}fps.mp4"
        )
    regular, bold = FONT_FAMILIES[args.font]
    args.equation_face = FONT_FAMILIES[args.equation_font][0]
    for face in (regular, bold, args.equation_face):
        if not Path(face).exists():
            raise RuntimeError(f"Missing font: {face}")
    glow.MONO_FONT = regular
    glow.MONO_BOLD_FONT = bold
    return args


if __name__ == "__main__":
    parsed = parse_args()
    print(f"Rendering {parsed.duration:g}s at {parsed.fps} fps -> {parsed.output}")
    print(f"Saved: {render(parsed).resolve()}")
