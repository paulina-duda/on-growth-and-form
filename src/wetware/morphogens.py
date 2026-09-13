#!/usr/bin/env python3
"""Three morphogenesis processes, each one an algorithm that is also a biology.

The common thread is not a look, it is a claim: that form can be *computed*.
Each of these is a rule simple enough to write in a few lines and rich enough
that a living thing appears to be running it.

* `Phyllotaxis` — a shoot apex placing each organ where the ones already there
  object least. No angle appears in the rule; the divergence and the Fibonacci
  spiral counts are what falls out of it.
* `Stripe` — a zebrafish pattern computed by the tissue rather than painted
  onto it, with pigment cells standing where Turing put two chemicals.
* `Tear` — a placozoan crawling itself apart. Every cell walks on its own and
  turns toward the pull of its neighbours; that is enough to align thousands of
  them, and enough to tear the sheet when two regions disagree.

Everything here is vectorised over the whole population or grid; nothing steps
one cell at a time.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def _laplacian(field: np.ndarray) -> np.ndarray:
    """Five-point stencil on a wrapping grid."""
    return (
        np.roll(field, 1, 0)
        + np.roll(field, -1, 0)
        + np.roll(field, 1, 1)
        + np.roll(field, -1, 1)
        - 4.0 * field
    )


def _blur3(field: np.ndarray) -> np.ndarray:
    """3x3 mean on a wrapping grid, as four rolls rather than a convolution."""
    return (
        field
        + np.roll(field, 1, 0)
        + np.roll(field, -1, 0)
        + np.roll(field, 1, 1)
        + np.roll(field, -1, 1)
    ) * 0.2


def _capsule(x, y, a, b, radius):
    """Inside test for a thick line segment, used to draw the section."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length = dx * dx + dy * dy
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / max(length, 1e-9), 0.0, 1.0)
    return (x - (ax + t * dx)) ** 2 + (y - (ay + t * dy)) ** 2 <= radius * radius


def _element_stiffness(poisson: float = 0.3) -> np.ndarray:
    """Stiffness of a unit square bilinear element, unit modulus, plane stress."""
    k = np.array([
        1 / 2 - poisson / 6, 1 / 8 + poisson / 8, -1 / 4 - poisson / 12, -1 / 8 + 3 * poisson / 8,
        -1 / 4 + poisson / 12, -1 / 8 - poisson / 8, poisson / 6, 1 / 8 - 3 * poisson / 8,
    ])
    order = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ])
    return (k[order] / (1.0 - poisson * poisson)).astype(np.float64)


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

        Same quantity the other growth pieces are coloured by -- when it grew --
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


class Stripe:
    """Turing was right about the stripes. He was wrong about the morphogens.

    A zebrafish stripe is not two chemicals racing each other across a sheet.
    It is two kinds of cell -- black melanophores and yellow xanthophores --
    each one reading how many of the other kind are nearby and, if the answer
    is wrong, becoming the other kind. The interaction has the shape Turing
    needed: a cell is supported by its own kind close in and suppressed by its
    own kind further out, which is short-range activation and long-range
    inhibition with whole cells standing where the chemicals were. Nakamasu
    and colleagues measured the two ranges by killing cells with a laser and
    watching what grew back.

    So the pattern is computed by the tissue rather than painted onto it, and
    the machinery is visible: every pixel of a stripe is an individual animal
    cell that decided, and can still change its mind.

    **The fish does not stop growing, and that is the piece.** The two ranges
    are fixed -- they are the reach of one cell's processes, and a cell does
    not get longer because the fish does. A stripe therefore has a width it
    wants, and skin that keeps widening carries the existing stripes apart
    until the gap between them is wide enough to hold another one, at which
    point a new stripe nucleates in the middle of it. A zebrafish adds its
    adult stripes exactly this way, ventrally and dorsally, as it grows. There
    is no counter anywhere and nothing decides how many stripes to make: the
    number is whatever the skin's height divided by that fixed width comes to.

    Two consequences worth stating, because both cost a rebuild to find:

    * **The radius has to advance by a fixed amount per step, not a fixed
      fraction.** The stripe count goes as the radius, so only linear growth
      spreads the splitting evenly across a clip; exponential growth puts two
      thirds of it in the last quarter and the first half is a still.
    * **The long-range term has to be strong enough to break a stripe from
      the inside.** Under a weak one the pattern is stable to stretching and
      the stripes simply get fatter as the skin grows -- which is a picture of
      a disc being scaled up, not of a pattern being recomputed. Measured as
      boundary length over radius: flat means stretching, rising means
      splitting.
    """

    def __init__(
        self,
        height: int,
        width: int,
        radius: float | None = None,
        seed_radius: float = 220.0,
        per_cell: float = 6.0,
        divisor: int = 3,
        sigma_short: float = 4.6,
        sigma_long: float = 13.6,
        # Stripes run along the body axis because the ranges are not the same
        # in both directions. Straight parallel bands need a much stronger bias
        # than this and stop looking like an animal -- at 7 the disc reads as a
        # barcode. 3.5 keeps them horizontal and lets them wander.
        anisotropy: float = 3.5,
        weight: float = 1.70,
        rate: float = 0.30,
        growth: float | None = None,
        wander: float = 0.80,
        recency: float = 26.0,
        seed: int = 20260904,
    ) -> None:
        from scipy.ndimage import gaussian_filter

        self._gaussian = gaussian_filter
        self.height, self.width, self.divisor = height, width, divisor
        self.rows, self.columns = height // divisor, width // divisor
        self.limit = radius if radius is not None else 0.44 * min(height, width)
        self.radius = float(seed_radius)
        self.centre = np.array([width * 0.5, height * 0.5], dtype=np.float64)
        self.per_cell, self.rate, self.wander = per_cell, rate, wander
        self.sigma_short = (sigma_short, sigma_short * anisotropy)
        self.sigma_long = (sigma_long, sigma_long * anisotropy)
        self.weight, self.recency = weight, recency
        self.growth = growth if growth is not None else 0.34
        self.generator = np.random.default_rng(seed)

        count = int(math.pi * self.radius ** 2 / per_cell)
        span = self.radius * np.sqrt(self.generator.random(count))
        angle = self.generator.uniform(0.0, 2.0 * math.pi, count)
        self.x = (self.centre[0] + span * np.cos(angle)).astype(np.float32)
        self.y = (self.centre[1] + span * np.sin(angle)).astype(np.float32)
        self.kind = (self.generator.random(count) < 0.5).astype(np.int8)
        self.age = np.zeros(count, dtype=np.float32)
        self.switches = 0

    def _bins(self) -> tuple[np.ndarray, np.ndarray]:
        row = np.clip((self.y / self.divisor).astype(np.int32), 0, self.rows - 1)
        column = np.clip((self.x / self.divisor).astype(np.int32), 0, self.columns - 1)
        return row, column

    def drive(self) -> np.ndarray:
        """Short-range support for one's own kind, minus long-range suppression.

        Smoothed with `mode="nearest"`, never wrapped: the skin has an edge and
        a cell at the rim must not read the far side of the disc as a
        neighbour, or stripes stitch themselves across the black.
        """
        row, column = self._bins()
        field = np.zeros((self.rows, self.columns), dtype=np.float32)
        np.add.at(field, (row, column), np.where(self.kind == 1, 1.0, -1.0))
        short = self._gaussian(field, self.sigma_short, mode="nearest")
        long_range = self._gaussian(field, self.sigma_long, mode="nearest")
        return short - self.weight * long_range

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            if self.radius < self.limit:
                previous = self.radius
                self.radius = min(self.radius + self.growth, self.limit)
                scale = self.radius / previous
                self.x = (self.centre[0] + (self.x - self.centre[0]) * scale).astype(np.float32)
                self.y = (self.centre[1] + (self.y - self.centre[1]) * scale).astype(np.float32)

            drive = self.drive()
            row, column = self._bins()
            want = (drive[row, column] > 0).astype(np.int8)
            # Only a fraction of the skin is up for replacement on any step.
            # Switching every cell that disagrees at once makes the boundary a
            # hard line that snaps between frames instead of a border being
            # argued over, which is what the clip is of.
            hot = self.generator.random(len(self.x)) < self.rate
            flip = hot & (want != self.kind)
            self.switches += int(flip.sum())
            self.kind = np.where(flip, want, self.kind)
            self.age = np.where(flip, 0.0, np.minimum(self.age + 1.0, 4.0 * self.recency)).astype(np.float32)

            self.x += (self.wander * self.generator.standard_normal(len(self.x))).astype(np.float32)
            self.y += (self.wander * self.generator.standard_normal(len(self.y))).astype(np.float32)

            # New skin is new cells, not existing ones spread thinner: hold the
            # areal density and let the population follow the area.
            target = int(math.pi * self.radius ** 2 / self.per_cell)
            missing = target - len(self.x)
            if missing > 0:
                span = self.radius * np.sqrt(self.generator.random(missing))
                angle = self.generator.uniform(0.0, 2.0 * math.pi, missing)
                born_x = (self.centre[0] + span * np.cos(angle)).astype(np.float32)
                born_y = (self.centre[1] + span * np.sin(angle)).astype(np.float32)
                born_row = np.clip((born_y / self.divisor).astype(np.int32), 0, self.rows - 1)
                born_column = np.clip((born_x / self.divisor).astype(np.int32), 0, self.columns - 1)
                self.x = np.concatenate((self.x, born_x))
                self.y = np.concatenate((self.y, born_y))
                self.kind = np.concatenate((self.kind, (drive[born_row, born_column] > 0).astype(np.int8)))
                self.age = np.concatenate((self.age, np.zeros(missing, dtype=np.float32)))

            # Confine the model, not the render: a cell pushed past the margin
            # is put back on it, so the black outside is black because nothing
            # was ever allowed to be there.
            offset = np.hypot(self.x - self.centre[0], self.y - self.centre[1])
            outside = offset > self.radius
            if outside.any():
                scale = self.radius / np.maximum(offset[outside], 1e-6)
                self.x[outside] = (self.centre[0] + (self.x[outside] - self.centre[0]) * scale).astype(np.float32)
                self.y[outside] = (self.centre[1] + (self.y[outside] - self.centre[1]) * scale).astype(np.float32)

    def boundary(self) -> int:
        """Stripe border on screen, in pixels. The quantity the piece is about.

        Divided by the radius it says which of the two things is happening: a
        flat ratio is stripes being stretched, a rising one is stripes being
        split.
        """
        drive = self.drive()
        grid_y, grid_x = np.mgrid[: self.rows, : self.columns]
        inside = (
            (grid_y - self.centre[1] / self.divisor) ** 2
            + (grid_x - self.centre[0] / self.divisor) ** 2
        ) < (self.radius / self.divisor - 2.0) ** 2
        warm = (drive > 0) & inside
        total = int((warm[:-1, :] != warm[1:, :])[inside[:-1, :] & inside[1:, :]].sum())
        total += int((warm[:, :-1] != warm[:, 1:])[inside[:, :-1] & inside[:, 1:]].sum())
        return total * self.divisor

    @property
    def count(self) -> int:
        return len(self.x)

    def species(self) -> np.ndarray:
        return self.kind

    def weights(self) -> np.ndarray:
        """A cell that has just changed its mind is worth more light.

        Density is otherwise flat everywhere inside the disc -- two-valued
        cells at one areal density give the log-density map a solid block to
        work on and the frame reads as a printed pattern rather than a
        population. Recency is the one thing about a cell that is neither its
        type nor its position, it is intrinsic, and it is concentrated exactly
        where the pattern is being decided.
        """
        return (1.0 + 2.4 * np.exp(-self.age / self.recency)).astype(np.float32)

    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions, and how recently each cell last changed type."""
        points = np.column_stack((self.x, self.y)).astype(np.float32)
        return points, np.exp(-self.age / self.recency).astype(np.float32)


def _halton(index: int, base: int) -> float:
    """One coordinate of a Halton point -- evenly spread without a lattice's rows."""
    f, out, i = 1.0, 0.0, index + 1
    while i > 0:
        f /= base
        out += f * (i % base)
        i //= base
    return out


class Tear:
    """Trichoplax adhaerens: a flat animal with no nerves that tears itself in two.

    A placozoan is two epithelia and a layer of fibre cells between them -- no
    nerves, no muscle, no organs -- and it crawls on a carpet of cilia. Every
    ventral cell walks on its own and nothing tells it which way the animal is
    going. Cells are tied to their neighbours, so a cell that walks is pulled by
    the ones around it, and it turns toward the pull. That one rule is enough to
    line thousands of them up into a single heading (Ferrante et al. 2013).
    Prakash, Bull & Prakash 2022 showed placozoan tissue tearing under its own
    crawling -- motility-induced fracture -- which is what this reproduces; the
    model is the Ferrante coupling with bonds that yield, not theirs.

    Because nothing coordinates the walking, a large enough animal holds
    patches that agree inside and disagree with each other, and the tissue
    between them stretches. Bonds past a strain give; bonds re-form between
    close cells of one animal, so the tissue yields instead of shattering; and
    an animal stretched far enough comes apart in two. That is how Trichoplax
    reproduces. Cells divide in place, so the halves grow back to the size at
    which they tear again -- growth is the clock. Without it the process is a
    relaxation: at the gate sixteen animals shattered into thirty-seven pieces
    in the first quarter and nothing tore after (85.2% / 0.0%).

    Colour is how stretched a cell is -- the mean strain on its bonds, smoothed
    -- so a line the tissue is about to open along lights up before it opens.

    Model units put one cell spacing at 1; the dish is `units` spacings across
    its radius and sits at `dish` px on screen.
    """

    def __init__(
        self,
        height: float = 1920.0,
        width: float = 1080.0,
        seed: int = 1,
        animals: int = 10,
        size: float = 7.0,
        dish: float = 475.0,
        units: float = 93.0,
        v0: float = 0.3,
        beta: float = 1.0,
        rot_noise: float = 0.02,
        spring: float = 1.0,
        repel: float = 3.0,
        yield_strain: float = 0.35,
        heal: float = 0.05,
        grow: float = 0.007,
        wall: float = 2.0,
        world: str = "dish",
        box: tuple = (108.0, 192.0),
        strain_ref: float = 0.13,
        min_draw: int = 20,
        dim: float = 0.3,
        layout: str = "random",
        dt: float = 0.05,
        heading_sd: float = 0.4,
        smooth: float = 0.05,
        spread: int = 2,
        trail_stride: int = 20,
        trail_points: int = 6,
        max_cells: int = 30_000,
        minsize: int = 40,
    ) -> None:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import Delaunay

        self._coo, self._components = coo_matrix, connected_components
        self.h, self.w = float(height), float(width)
        self.R = float(units)
        # In a wrapping world the box has to tile the frame exactly, or the
        # seam shows as a margin down the side; the dish instead sets its own
        # scale from its radius.
        self.scale = (float(width) / float(box[0])) if world == "field" else float(dish) / self.R
        self.rng = np.random.default_rng(seed)
        self.v0, self.beta, self.rot_noise = v0, beta, rot_noise
        self.spring, self.repel, self.yield_strain = spring, repel, yield_strain
        self.heal_p, self.grow, self.wall = heal, grow, wall
        # `world="field"`: no dish and no walls, the frame wrapping on itself.
        # Self-propelled walkers pile up against any wall -- the dish cut ended
        # as a ring of animals round an empty middle, 9.7% of cells within two
        # spacings of the rim -- and a wrapping world has nothing to pile on.
        self.world = world
        self.box = np.asarray(box, float)
        self.strain_ref, self.min_draw, self.dim = strain_ref, min_draw, dim
        self.dt, self.smooth, self.spread = dt, smooth, spread
        self.max_cells, self.minsize = max_cells, minsize
        self.step_index = 0
        self.breaks = self.heals = self.divisions = 0

        # Animals as wobbly discs on a jittered hex lattice, kept apart and
        # inside the dish. The outline wobbles because a real placozoan is never
        # round, and a lattice disc tears along its lattice rows.
        rng = self.rng
        centres: list[np.ndarray] = []
        tries = 0
        while len(centres) < animals and tries < 20_000:
            tries += 1
            if self.world == "field":
                # `layout="spread"`: a low-discrepancy pair (Halton 2, 3) with a
                # minimum separation, so the animals start spread over the whole
                # frame. Drawing the starts at random leaves holes -- the first
                # field cut spent much of its length with an empty middle -- and
                # in a wrapping world a hole does not fill in from the edges.
                if layout == "spread":
                    k = tries + int(seed) * 7
                    c = np.array([_halton(k, 2), _halton(k, 3)]) * self.box
                else:
                    c = rng.uniform(0.0, 1.0, 2) * self.box
                if all(np.hypot(*self._delta(c - q)) > 2.6 * size for q in centres):
                    centres.append(c)
                continue
            c = rng.uniform(-self.R, self.R, 2)
            if np.hypot(*c) > self.R - 1.6 * size:
                continue
            if all(np.hypot(*(c - q)) > 2.6 * size for q in centres):
                centres.append(c)
        pts, bonds, heading, base = [], [], [], 0
        g = np.arange(-size - 2, size + 3)
        for c in centres:
            X, Y = np.meshgrid(g, g * 0.866)
            X = X + 0.5 * (np.round(Y / 0.866) % 2)
            p = np.column_stack((X.ravel(), Y.ravel()))
            ph = np.arctan2(p[:, 1], p[:, 0])
            wob = (1.0 + 0.12 * np.cos(2 * ph + rng.uniform(0, 2 * np.pi))
                   + 0.06 * np.cos(3 * ph + rng.uniform(0, 2 * np.pi)))
            p = p[np.hypot(p[:, 0], p[:, 1]) < size * wob]
            p = p + rng.normal(0.0, 0.08, p.shape)
            tri = Delaunay(p)
            e = np.concatenate([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]])
            e = np.unique(np.sort(e, axis=1), axis=0)
            e = e[np.hypot(*(p[e[:, 0]] - p[e[:, 1]]).T) < 1.4]
            pts.append(p + c)
            bonds.append(e + base)
            # One heading per animal with a spread: an animal that starts in
            # random disorder shatters on the first step, which is the gate's
            # relaxation again.
            heading.append(rng.uniform(0, 2 * np.pi) + rng.normal(0.0, heading_sd, len(p)))
            base += len(p)
        self.pos = np.concatenate(pts)
        self.theta = np.concatenate(heading)
        self.bi, self.bj = np.concatenate(bonds).T
        self.l0 = np.hypot(*(self.pos[self.bj] - self.pos[self.bi]).T)
        self.strain = np.zeros(len(self.pos))
        self.lab = self._labels()
        # Which animal a cell belongs to, and that animal's colour. A tear
        # gives the smaller half a tint a third of the way round the ramp from
        # its parent's, so one body coming apart reads as one colour becoming
        # two rather than as a shape changing.
        self.body = self.lab.copy()
        self._next_body = int(self.body.max()) + 1
        self.tint = (self.body * 0.381966) % 1.0
        self.splits = self.flakes = 0
        here = self._px()
        self.trail_stride, self.trail_points = trail_stride, trail_points
        self.history = np.repeat(here[None, :, :], trail_points, axis=0)
        self.trail_phase = 0.0

    # ----------------------------------------------------------- mechanics
    def _delta(self, v: np.ndarray) -> np.ndarray:
        """Shortest way round, when the world wraps."""
        if self.world != "field":
            return v
        return v - self.box * np.round(v / self.box)

    def _px(self) -> np.ndarray:
        if self.world == "field":
            return np.column_stack((self.pos[:, 0] * self.scale,
                                    self.h - self.pos[:, 1] * self.scale)).astype(np.float32)
        return np.column_stack((self.w * 0.5 + self.pos[:, 0] * self.scale,
                                self.h * 0.5 - self.pos[:, 1] * self.scale)).astype(np.float32)

    def _labels(self) -> np.ndarray:
        n = len(self.pos)
        g = self._coo((np.ones(len(self.bi)), (self.bi, self.bj)), shape=(n, n))
        return self._components(g, directed=False)[1]

    def _one(self) -> None:
        pos, n = self.pos, len(self.pos)
        F = np.zeros((n, 2))
        if self.world == "field":
            wrapped = np.clip(np.mod(pos, self.box), 0.0, self.box - 1e-9)
            pairs = cKDTree(wrapped, boxsize=self.box).query_pairs(1.0, output_type="ndarray")
        else:
            pairs = cKDTree(pos).query_pairs(1.0, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            dv = self._delta(pos[j] - pos[i])
            d = np.hypot(dv[:, 0], dv[:, 1])
            fr = (self.repel * (1.0 - d) / np.maximum(d, 1e-9))[:, None] * dv
            np.add.at(F, i, -fr)
            np.add.at(F, j, fr)
        dv = self._delta(pos[self.bj] - pos[self.bi])
        d = np.hypot(dv[:, 0], dv[:, 1])
        fb = (self.spring * (d - self.l0) / np.maximum(d, 1e-9))[:, None] * dv
        np.add.at(F, self.bi, fb)
        np.add.at(F, self.bj, -fb)
        # Strain per cell: the mean stretch on its bonds, compression counted
        # as zero, then averaged with its bonded neighbours `spread` times and
        # smoothed over about a frame. Both averages are there for the colour.
        # Raw in time it flickers bond by bond -- `culture`'s blinking dots --
        # and raw in space it is confetti: the first render lit 0.96% of the
        # dish as isolated sparks, where a tear opens along a line.
        stretch = np.maximum((d - self.l0) / self.l0, 0.0)
        total = np.bincount(self.bi, stretch, n) + np.bincount(self.bj, stretch, n)
        count = np.bincount(self.bi, minlength=n) + np.bincount(self.bj, minlength=n)
        local = total / np.maximum(count, 1)
        for _ in range(self.spread):
            local = (local + np.bincount(self.bi, local[self.bj], n)
                     + np.bincount(self.bj, local[self.bi], n)) / (1.0 + count)
        self.strain += self.smooth * (local - self.strain)
        keep = (d - self.l0) / self.l0 < self.yield_strain
        self.breaks += int((~keep).sum())
        self.bi, self.bj, self.l0 = self.bi[keep], self.bj[keep], self.l0[keep]

        # The dish: a spring that switches on only past the rim, which is the
        # reel skill's answer for anything with a velocity. It is in F, so a
        # cell turns away from the wall the same way it turns toward any pull.
        if self.world != "field":
            r = np.hypot(pos[:, 0], pos[:, 1])
            out = r > self.R
            if out.any():
                F[out] -= (self.wall * (r[out] - self.R) / r[out])[:, None] * pos[out]

        if self.step_index % 10 == 0:
            self.lab = self._labels()
            self._reconcile()
            if len(pairs) and self.heal_p > 0:
                i, j = pairs[:, 0], pairs[:, 1]
                dv = self._delta(pos[j] - pos[i])
                close = (np.hypot(dv[:, 0], dv[:, 1]) < 1.05) & (self.lab[i] == self.lab[j])
                code = np.minimum(i, j) * n + np.maximum(i, j)
                have = np.minimum(self.bi, self.bj) * n + np.maximum(self.bi, self.bj)
                new = close & ~np.isin(code, have) & (self.rng.random(len(i)) < self.heal_p)
                if new.any():
                    self.bi = np.concatenate([self.bi, i[new]])
                    self.bj = np.concatenate([self.bj, j[new]])
                    self.l0 = np.concatenate([self.l0, np.maximum(np.hypot(*dv[new].T), 0.9)])
                    self.heals += int(new.sum())

        # Turn toward the pull (Ferrante 2013), then walk.
        mag = np.hypot(F[:, 0], F[:, 1])
        self.theta += (self.dt * self.beta * mag * np.sin(np.arctan2(F[:, 1], F[:, 0]) - self.theta)
                       + math.sqrt(2.0 * self.rot_noise * self.dt) * self.rng.standard_normal(n))
        vel = self.v0 * np.column_stack((np.cos(self.theta), np.sin(self.theta))) + F
        self.pos = pos + self.dt * vel
        if self.world == "field":
            self.pos = np.mod(self.pos, self.box)

        # The clock: a cell divides in place, its daughter bonded to it at a
        # rest length it has not reached yet, so the sheet is pushed outward.
        if self.grow > 0 and n < self.max_cells:
            div = np.flatnonzero(self.rng.random(n) < self.grow * self.dt)
            if len(div):
                a = self.rng.uniform(0, 2 * np.pi, len(div))
                off = 0.25 * np.column_stack((np.cos(a), np.sin(a)))
                born = self.pos[div] + off
                self.pos[div] -= off
                new = np.arange(n, n + len(div))
                self.pos = np.concatenate([self.pos, born])
                self.theta = np.concatenate([self.theta, self.theta[div]])
                self.strain = np.concatenate([self.strain, self.strain[div]])
                self.lab = np.concatenate([self.lab, self.lab[div]])
                self.body = np.concatenate([self.body, self.body[div]])
                self.tint = np.concatenate([self.tint, self.tint[div]])
                self.bi = np.concatenate([self.bi, div])
                self.bj = np.concatenate([self.bj, new])
                self.l0 = np.concatenate([self.l0, np.ones(len(div))])
                here = self._px()[new]
                self.history = np.concatenate([self.history, np.repeat(here[None, :, :], self.trail_points, 0)], 1)
                self.divisions += len(div)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self._one()
            self.step_index += 1
            if self.step_index % self.trail_stride == 0:
                self.history[:-1] = self.history[1:]
                self.history[-1] = self._px()
            self.trail_phase = (self.step_index % self.trail_stride) / self.trail_stride

    def _reconcile(self) -> None:
        """Give every piece of a body that has come apart its own identity.

        The biggest piece keeps the parent's; each other piece over `min_draw`
        cells is a tear and takes a new tint, and anything smaller is a flake
        shed off an edge. Counted separately, because a body splitting in two
        is the event and a shed flake is not.
        """
        order = np.argsort(self.lab, kind="stable")
        bounds = np.flatnonzero(np.diff(self.lab[order])) + 1
        for cells in np.split(order, bounds):
            parents, counts = np.unique(self.body[cells], return_counts=True)
            keep = parents[counts.argmax()]
            self.body[cells] = keep
        for parent in np.unique(self.body):
            mine = self.body == parent
            comps, sizes = np.unique(self.lab[mine], return_counts=True)
            if len(comps) < 2:
                continue
            biggest = comps[sizes.argmax()]
            for comp, size in zip(comps, sizes):
                if comp == biggest:
                    continue
                piece = mine & (self.lab == comp)
                self.body[piece] = self._next_body
                self.tint[piece] = (self.tint[piece][0] + 0.381966) % 1.0
                self._next_body += 1
                if size >= self.min_draw:
                    self.splits += 1
                else:
                    self.flakes += 1

    def _draw_weight(self) -> np.ndarray:
        """Brightness: stretch, with anything too small to be an animal kept dim."""
        lit = 0.55 + 0.45 * np.clip(self.strain / self.strain_ref, 0.0, 1.0)
        sizes = np.bincount(self.lab)[self.lab]
        return (np.where(sizes >= self.min_draw, lit, self.dim * lit)).astype(np.float32)

    # -------------------------------------------------------------- probes
    @property
    def count(self) -> int:
        return len(self.pos)

    def animals(self) -> int:
        """Bodies of at least `minsize` cells -- a shed flake is not an animal."""
        sizes = np.bincount(self._labels())
        return int((sizes >= self.minsize).sum())

    def density(self) -> np.ndarray:
        grid, _, _ = np.histogram2d(self.pos[:, 0], self.pos[:, 1], bins=64,
                                    range=[[-self.R, self.R], [-self.R, self.R]])
        return grid

    def trails(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each cell's recent path, how stretched it is, and a weight of 1.

        Same shape and the same sliding offset as the other trail models; strain
        goes out raw and the renderer scales it against the cover's 98th
        percentile, because it is bimodal -- most cells sit near zero -- and
        ranking would drag the relaxed tissue up the ramp.
        """
        here = self._px()
        older, newer = self.history[:-1], self.history[1:]
        slid = (older + self.trail_phase * (newer - older)).astype(np.float32)
        path = np.concatenate((slid, here[None, :, :]), axis=0)
        return path, self.tint.astype(np.float32), self._draw_weight()

    def swarm(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._px(), self.tint.astype(np.float32), self._draw_weight()

