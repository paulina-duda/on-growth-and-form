#!/usr/bin/env python3
"""Lenia: Conway's Life with the corners taken off, and animals in it.

Conway's rule counts eight neighbours, compares the count to two integers, and
switches a cell on or off. Lenia (Bert Wang-Chak Chan, 2019) keeps the shape of
that idea and makes every part of it continuous. A cell holds a real number
between 0 and 1 instead of a bit. Its neighbourhood is a smooth ring rather than
a square of eight, and "how many neighbours" becomes an average over that ring,
weighted by a kernel. The birth and survival intervals become one smooth bump:
a growth curve that returns how much to add or subtract, peaking at some
neighbourhood weight `mu` and falling off with width `sigma`. And time is
divided, so each step moves the field by a tenth of the growth instead of all of
it.

Four changes, none of them clever, and what comes out are not blinkers and
gliders but *solitons*: lumps of continuous field, smooth-edged and internally
structured, that hold themselves together while they travel. They have fronts
and wakes. They deform when they pass each other. Some of them are stable for as
long as you care to run them; most possible ones dissolve in a hundred steps or
explode until they fill the world.

Which is why nothing here is designed. A creature in Lenia is a point in a space
of parameters and initial conditions, most of that space is lethal, and the way
anyone finds an animal in it -- Chan included -- is to look.
"""

from __future__ import annotations

import numpy as np

try:  # the search is a few hundred short simulations; a GPU makes it minutes
    import torch
except ImportError:  # pragma: no cover
    torch = None


def bump(x: np.ndarray) -> np.ndarray:
    """exp(4 - 1/(x(1-x))) on (0, 1), zero outside: Lenia's standard smooth ring.

    It rises from nothing at 0, peaks at 1/2 and returns to nothing at 1, with
    every derivative vanishing at both ends -- which is what keeps the kernel
    from having an edge for the field to catch on.
    """
    inside = (x > 0.0) & (x < 1.0)
    safe = np.where(inside, x, 0.5)
    return np.where(inside, np.exp(4.0 - 1.0 / (safe * (1.0 - safe))), 0.0)


def ring_kernel(
    height: int, width: int, radius: float, beta: tuple[float, ...] = (1.0,)
) -> np.ndarray:
    """The neighbourhood: concentric smooth rings, normalised to sum to one.

    The radial coordinate is cut into `len(beta)` bands and the same bump is put
    into each, scaled by its weight. `beta = (1,)` is a single ring, which is the
    neighbourhood Orbium lives in; two or three of different weights give the
    field more to say and tend to hold more elaborate creatures.

    Laid out for the whole field rather than for a patch, because the step is a
    convolution done in Fourier space and the kernel has to be the same shape as
    the thing it multiplies.
    """
    rows = np.fft.fftfreq(height, d=1.0 / height)
    columns = np.fft.fftfreq(width, d=1.0 / width)
    distance = np.hypot(*np.meshgrid(rows, columns, indexing="ij")) / radius

    rings = len(beta)
    band = np.minimum((distance * rings).astype(int), rings - 1)
    weights = np.asarray(beta, dtype=np.float64)[band]
    kernel = weights * bump((distance * rings) % 1.0)
    kernel[distance >= 1.0] = 0.0
    return kernel / kernel.sum()


class Lenia:
    """A continuous automaton: one ring kernel, one growth curve, divided time.

    `radius` is the kernel's reach in cells and sets how large a creature is;
    `mu` and `sigma` are the growth curve, and between them they are most of
    what distinguishes one animal from another. `beta` gives the kernel its
    rings -- a single ring is the classic Orbium neighbourhood, and two or three
    concentric rings of different weights give the field more to say and tend to
    hold more elaborate creatures.

    The neighbourhood average is a convolution, and a convolution over a whole
    field is a multiplication in Fourier space. The kernel is transformed once
    at construction; after that a step is two transforms and some arithmetic,
    whatever the radius is -- which is the only reason a kernel 26 cells across
    is affordable at this frame size.
    """

    def __init__(
        self,
        height: int,
        width: int,
        radius: float = 18.0,
        mu: float = 0.15,
        sigma: float = 0.017,
        beta: tuple[float, ...] = (1.0,),
        steps_per_time: float = 10.0,
        device: str | None = None,
    ) -> None:
        self.height, self.width = height, width
        self.radius, self.mu, self.sigma = radius, mu, sigma
        self.dt = 1.0 / steps_per_time
        if device is None:
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.device = device if torch is not None else "cpu"

        kernel = ring_kernel(height, width, radius, beta)

        self.field = np.zeros((height, width), dtype=np.float32)
        self.growth = np.zeros((height, width), dtype=np.float32)
        self.step_index = 0
        if self.device != "cpu":
            self._kernel = torch.fft.rfft2(
                torch.tensor(kernel, dtype=torch.float32, device=self.device)
            )
            self._field = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        else:
            self._kernel = np.fft.rfft2(kernel)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def place(self, patch: np.ndarray, row: int, column: int) -> None:
        """Drop a patch into the field, wrapping at the edges."""
        rows = (np.arange(patch.shape[0]) + row) % self.height
        columns = (np.arange(patch.shape[1]) + column) % self.width
        self.field[np.ix_(rows, columns)] = np.maximum(
            self.field[np.ix_(rows, columns)], patch.astype(np.float32)
        )
        if self.device != "cpu":
            self._field = torch.tensor(self.field, dtype=torch.float32, device=self.device)

    def clear(self) -> None:
        self.field[:] = 0.0
        if self.device != "cpu":
            self._field.zero_()

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _growth_curve(self, potential):
        if self.device != "cpu":
            return 2.0 * torch.exp(-(((potential - self.mu) / self.sigma) ** 2) / 2.0) - 1.0
        return 2.0 * np.exp(-(((potential - self.mu) / self.sigma) ** 2) / 2.0) - 1.0

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.step_index += 1
            if self.device != "cpu":
                potential = torch.fft.irfft2(
                    torch.fft.rfft2(self._field) * self._kernel, s=(self.height, self.width)
                )
                growth = self._growth_curve(potential)
                self._field = torch.clamp(self._field + self.dt * growth, 0.0, 1.0)
            else:
                potential = np.fft.irfft2(
                    np.fft.rfft2(self.field) * self._kernel, s=(self.height, self.width)
                )
                growth = self._growth_curve(potential).astype(np.float32)
                self.field = np.clip(self.field + self.dt * growth, 0.0, 1.0)
        self._sync(growth)

    def _sync(self, growth) -> None:
        if self.device != "cpu":
            self.field = self._field.detach().cpu().numpy()
            self.growth = growth.detach().cpu().numpy()
        else:
            self.growth = np.asarray(growth, dtype=np.float32)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def mass(self) -> float:
        return float(self.field.sum())

    def centre(self) -> tuple[float, float]:
        """Centre of mass on a torus, via the circular mean of each axis.

        A creature sitting on the seam has half its mass at row 2 and half at
        row 510, and an arithmetic mean puts its centre in the middle of the
        field where there is nothing at all. Averaging the angle instead is the
        only way to ask this question of a wrapped world.
        """
        total = max(self.mass(), 1e-9)
        centres = []
        for axis, length in ((1, self.height), (0, self.width)):
            profile = self.field.sum(axis=axis)
            angle = np.arange(length) * (2.0 * np.pi / length)
            mean = np.arctan2(
                float((profile * np.sin(angle)).sum() / total),
                float((profile * np.cos(angle)).sum() / total),
            )
            centres.append(float(mean % (2.0 * np.pi)) * length / (2.0 * np.pi))
        return centres[0], centres[1]

    def spread(self) -> float:
        """Radius of gyration about the wrapped centre of mass, in cells."""
        total = max(self.mass(), 1e-9)
        row, column = self.centre()
        rows = np.abs(np.arange(self.height) - row)
        columns = np.abs(np.arange(self.width) - column)
        rows = np.minimum(rows, self.height - rows)
        columns = np.minimum(columns, self.width - columns)
        distance = np.hypot(rows[:, None], columns[None, :])
        return float(np.sqrt((self.field * distance**2).sum() / total))

    def fields(self) -> tuple[np.ndarray, np.ndarray]:
        """Density from how much field is here, colour from whether it is growing.

        The state alone makes a smooth lump and says nothing about which way it
        is going. The growth term does: it is strongly positive along the edge
        the creature is advancing into and negative through the wake it is
        leaving, so colouring by it separates the front of an animal from its
        back -- which is the whole difference between a picture of a thing and a
        picture of a thing *moving*.
        """
        density = self.field.astype(np.float32)
        shade = np.clip(0.5 + 0.5 * self.growth, 0.0, 1.0).astype(np.float32)
        return density, shade


def crescent(
    radius: float,
    ring: float = 0.5,
    thickness: float = 0.2,
    lobe: float = 2.0,
    phase: float = 0.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    """An arc: a soft annulus with one side faded out.

    Every travelling creature anyone has found in Lenia looks like this at the
    start, and the reason is not aesthetic. A symmetric seed has no direction to
    pick and either dies or grows evenly outward; a lump with a thick side and a
    thin one has a front and a back on the first step, and a front and a back is
    all a soliton needs to start going somewhere. `phase` is which way it faces,
    which makes it also the direction the animal will swim.

    Six numbers, so a creature found by the search can be written down and
    rebuilt exactly rather than shipped as an array of pixels.
    """
    size = int(radius * 2.8)
    centre = size / 2.0
    rows, columns = np.mgrid[0:size, 0:size].astype(np.float64)
    distance = np.hypot(rows - centre, columns - centre) / radius
    angle = np.arctan2(rows - centre, columns - centre)
    shell = np.exp(-((distance - ring) ** 2) / (2.0 * thickness**2))
    side = np.clip(np.cos(angle - phase) * 0.5 + 0.5, 0.0, 1.0) ** lobe
    return np.clip(shell * side * amplitude, 0.0, 1.0)


def random_seed(generator: np.random.Generator, radius: float, blobs: int = 5) -> np.ndarray:
    """A small patch of smooth lumps -- the only thing the search gets to vary.

    Structured noise rather than white noise: a field of independent random
    cells has nothing at the kernel's length scale and dissolves on the first
    step whatever the parameters are, so every candidate would score the same
    and the search would be measuring nothing.
    """
    size = int(radius * 2.6)
    rows, columns = np.mgrid[0:size, 0:size].astype(np.float64)
    patch = np.zeros((size, size), dtype=np.float64)
    for _ in range(blobs):
        centre_row, centre_column = generator.uniform(size * 0.25, size * 0.75, 2)
        width = generator.uniform(radius * 0.25, radius * 0.6)
        amplitude = generator.uniform(0.4, 1.0)
        patch += amplitude * np.exp(
            -((rows - centre_row) ** 2 + (columns - centre_column) ** 2) / (2.0 * width**2)
        )
    return np.clip(patch, 0.0, 1.0)


def audition(
    radius: float = 30.0,
    candidates: int = 900,
    settle: int = 300,
    hold: int = 600,
    domain: int = 288,
    seed: int = 17,
    mu_range: tuple[float, float] = (0.10, 0.30),
    sigma_range: tuple[float, float] = (0.012, 0.042),
    betas: tuple[tuple[float, ...], ...] = ((1.0,), (1.0, 0.66), (1.0, 0.5, 0.25)),
    report=None,
) -> list[dict]:
    """Try random creatures and keep the ones that are still an individual.

    Each candidate is a growth curve drawn from `mu_range` × `sigma_range`, a
    kernel from `betas`, and an arc to start from. The defaults are the whole
    region where Lenia is worth looking at; narrowing them is how the second
    pass of a search works, once a first pass has shown where the survivors are. It
    runs alone in a small world and is then asked three things in order.

    **Is it still alive?** Most are not. Lenia has two failure modes and both
    are common: the field collapses to nothing, or it grows without limit until
    it fills the world. Roughly one seed in a hundred lands between them.

    **Is it still an individual?** A creature holds its own mass steady --
    within a factor of the mass it had three hundred steps ago -- and keeps it
    within a couple of kernel radii of its own centre. A pattern that has smeared
    into a texture fails both tests, and a colony that is still slowly spreading
    fails the first, which is exactly what a shorter audition lets through.

    **Does it go anywhere?** How far its centre of mass travels, which is what
    separates a soliton from a stationary ring. It is worth most of the score
    but is not a veto: a thing that sits and pulses is still an animal.

    Returns every survivor, best first, each described entirely by numbers --
    the growth curve, the kernel rings and the six parameters of its arc -- so
    the piece can rebuild it exactly without shipping a pixel of it.
    """
    generator = np.random.default_rng(seed)
    survivors: list[dict] = []
    for index in range(candidates):
        beta = betas[index % len(betas)]
        recipe = {
            "mu": float(generator.uniform(*mu_range)),
            "sigma": float(generator.uniform(*sigma_range)),
            "beta": beta,
            "ring": float(generator.uniform(0.35, 0.65)),
            "thickness": float(generator.uniform(0.12, 0.30)),
            "lobe": float(generator.uniform(1.4, 3.0)),
            "phase": float(generator.uniform(0.0, 2.0 * np.pi)),
            "amplitude": float(generator.uniform(0.5, 1.0)),
        }
        patch = crescent(
            radius, recipe["ring"], recipe["thickness"], recipe["lobe"],
            recipe["phase"], recipe["amplitude"],
        )
        world = Lenia(
            domain, domain, radius=radius,
            mu=recipe["mu"], sigma=recipe["sigma"], beta=recipe["beta"],
        )
        world.place(patch, domain // 2 - patch.shape[0] // 2, domain // 2 - patch.shape[1] // 2)

        world.step(settle)
        settled_mass, settled_centre = world.mass(), world.centre()
        if not 200.0 < settled_mass < 0.06 * domain * domain:
            continue
        world.step(hold)
        mass, spread = world.mass(), world.spread()
        if not (0.5 < mass / max(settled_mass, 1e-9) < 1.6 and spread < radius * 2.6):
            continue

        row, column = world.centre()
        travel = float(
            np.hypot(
                min(abs(row - settled_centre[0]), domain - abs(row - settled_centre[0])),
                min(abs(column - settled_centre[1]), domain - abs(column - settled_centre[1])),
            )
        )
        recipe.update(
            radius=radius, mass=float(mass), spread=float(spread), travel=travel,
            score=travel / radius + 0.3 * (radius * 2.6 - spread) / radius,
        )
        survivors.append(recipe)
        if report is not None:
            report(
                f"  survivor {len(survivors):2d} of {index + 1:4d} tried: "
                f"rings {len(beta)} mu {recipe['mu']:.3f} sigma {recipe['sigma']:.3f} "
                f"travelled {travel:5.1f} spread {spread:4.1f} -> {recipe['score']:5.2f}"
            )
    survivors.sort(key=lambda entry: entry["score"], reverse=True)
    return survivors


def confirm(recipe: dict, radius: float, domain: int, settle: int = 300, hold: int = 600) -> dict | None:
    """Re-run a found creature at the size it will actually be filmed at.

    Lenia is only approximately scale-invariant. The same growth curve and the
    same arc, drawn twice as large, is a different discretisation of the same
    equations, and creatures do not all survive the change: of four solitons
    found at radius 18, three carried over to radius 30 unchanged and the fourth
    stopped being self-limiting and grew until it filled the world. So nothing
    goes into a piece on the strength of the audition alone.
    """
    patch = crescent(
        radius, recipe["ring"], recipe["thickness"], recipe["lobe"],
        recipe["phase"], recipe["amplitude"],
    )
    world = Lenia(
        domain, domain, radius=radius,
        mu=recipe["mu"], sigma=recipe["sigma"], beta=recipe["beta"],
    )
    world.place(patch, domain // 2 - patch.shape[0] // 2, domain // 2 - patch.shape[1] // 2)
    world.step(settle)
    settled_mass, settled_centre = world.mass(), world.centre()
    if not 200.0 < settled_mass < 0.06 * domain * domain:
        return None
    world.step(hold)
    mass, spread = world.mass(), world.spread()
    if not (0.5 < mass / max(settled_mass, 1e-9) < 1.6 and spread < radius * 2.6):
        return None
    row, column = world.centre()
    travel = float(
        np.hypot(
            min(abs(row - settled_centre[0]), domain - abs(row - settled_centre[0])),
            min(abs(column - settled_centre[1]), domain - abs(column - settled_centre[1])),
        )
    )
    return dict(recipe, radius=radius, mass=float(mass), spread=float(spread), travel=travel)
