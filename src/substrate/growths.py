#!/usr/bin/env python3
"""Processes on a substrate.

The model published here is `Nematic`: microtubules, a motor that walks along
them, and the fuel it runs on, with no cell anywhere in it. It is the odd one
out of its edition twice over -- nothing in it is alive, and it adds no material
at all after the first step. It only rearranges what is already in the dish, and
spends the whole clip destroying its own order.

Every model exposes the same three things -- `step`, `metric` and whatever the
renderer needs to draw -- so the renderer can measure a process it knows nothing
about and schedule frames by how much the picture is actually changing.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

try:  # optional: Cahn-Hilliard wants a small timestep and many of them
    import torch
except ImportError:  # pragma: no cover
    torch = None


class Nematic:
    """An active nematic: microtubules, kinesin and ATP, and not one cell.

    Take microtubules out of a cell, add the motor that walks along them and
    the fuel that motor runs on, and let the mixture spread into a film one
    filament thick. Nothing in there is alive. There is no membrane, no
    genome, no signal, and nothing that could be said to be deciding
    anything. What the film does is tear itself apart, for ever.

    The filaments line up with their neighbours -- that is all "nematic"
    means, the alignment a liquid crystal has without any of the ordering a
    solid has. Kinesin walks along one filament carrying another, so
    neighbouring filaments slide past each other, and sliding along a
    direction pushes fluid along that direction. That is the whole rule:
    **alignment is turned into flow**. And the flow it produces bends the
    alignment that produced it, which bends the flow, and an aligned film
    turns out to be unstable to its own activity -- the *bend instability*,
    and there is no parameter at which it is not there.

    A bend that grows far enough cannot stay a bend. The director has to
    break, and where it breaks the film is left with a point around which
    the alignment turns by half a turn -- a topological defect, +1/2 or
    -1/2, and the halves are why the fabric cannot simply heal. Charge is
    conserved, so they are born in pairs; the +1/2 has a comet's head and
    swims, the -1/2 has three-fold symmetry and mostly sits; and when a +1/2
    finds a -1/2 they annihilate and that piece of film is whole again. The
    steady state is not order and not disorder but a fixed rate of tearing:
    active turbulence.

    Modelled the standard way -- Beris-Edwards for the alignment, one elastic
    constant, coupled to Stokes flow for the fluid, with an active stress
    proportional to the alignment itself, `sigma = -zeta Q`. Extensile
    filaments are `zeta > 0`.

    **The film is a drop, not a crop.** `radius` masks the *activity* and the
    ordering, not the drawing: outside it the Landau term has no well to sit
    in, so the alignment decays to isotropic and there is no film there to
    photograph. The fluid outside is still solved -- a real drop drags the
    bath around it.
    """

    def __init__(
        self,
        size: int,
        radius: float | None = None,
        elasticity: float = 0.04,
        landau: float = 1.0,
        rotation: float = 1.0,
        alignment: float = 0.7,
        activity: float = 0.030,
        viscosity: float = 1.0,
        dt: float = 0.05,
        seeded_order: float = 0.4,
        disorder: float = 0.05,
        afterglow: float = 150.0,
        edge: float = 3.0,
        seed: int = 20260903,
        device: str | None = None,
    ) -> None:
        self.size = size
        self.elasticity, self.landau = elasticity, landau
        self.rotation, self.alignment = rotation, alignment
        self.activity, self.viscosity, self.dt = activity, viscosity, dt
        self.radius = float(size * 0.44 if radius is None else radius)
        # The phosphor half-life, in steps. A defect is one pixel and it is
        # past in a moment; without a memory of where they have been the frame
        # is a texture with no events in it. Same device as `Excitable`.
        self.decay = float(0.5 ** (1.0 / max(afterglow, 1.0)))
        self.step_index = 0

        generator = np.random.default_rng(seed)
        rows, columns = np.mgrid[0:size, 0:size].astype(np.float32)
        distance = np.hypot(rows - (size - 1) / 2.0, columns - (size - 1) / 2.0)
        # Smooth, because a step in the mask is a step in the free energy and
        # the solver answers it with a ring of spurious order at the rim.
        self.mask = (0.5 * (1.0 - np.tanh((distance - self.radius) / edge))).astype(np.float32)
        self.dish = distance < self.radius

        # An aligned film, with just enough noise for the instability to have
        # something to grow from. Frame one is the turbulence; this is what the
        # clip cuts back to.
        angle = disorder * generator.standard_normal((size, size)).astype(np.float32)
        self.qxx = (seeded_order * np.cos(2.0 * angle) * self.mask).astype(np.float32)
        self.qxy = (seeded_order * np.sin(2.0 * angle) * self.mask).astype(np.float32)
        self.wake = np.zeros((size, size), dtype=np.float32)
        self.speed = np.zeros((size, size), dtype=np.float32)

        wave = 2.0 * np.pi * np.fft.fftfreq(size)
        kx, ky = np.meshgrid(wave, wave, indexing="ij")
        self.kx, self.ky = kx.astype(np.float32), ky.astype(np.float32)
        k2 = (kx * kx + ky * ky).astype(np.float32)
        self.k2 = k2
        safe = k2.copy()
        safe[0, 0] = 1.0
        self.k2_safe = safe

        if device is None:
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.device = device if torch is not None else "cpu"
        if self.device != "cpu":
            to = lambda a: torch.tensor(a, device=self.device)
            self._qxx, self._qxy = to(self.qxx), to(self.qxy)
            self._wake, self._speed = to(self.wake), to(self.speed)
            self._mask = to(self.mask)
            self._kx, self._ky = to(self.kx), to(self.ky)
            self._k2, self._k2_safe = to(self.k2), to(self.k2_safe)

    # ------------------------------------------------------------------
    # Spectral operators. Everything is periodic; the drop never touches the
    # box, so wrapping costs nothing and buys an exact Stokes solve.

    def _pack(self):
        if self.device != "cpu":
            return (torch, torch.fft, self._qxx, self._qxy, self._mask,
                    self._kx, self._ky, self._k2, self._k2_safe)
        return (np, np.fft, self.qxx, self.qxy, self.mask,
                self.kx, self.ky, self.k2, self.k2_safe)

    def _flow(self, qxx, qxy):
        """Stokes velocity driven by the active stress `sigma = -zeta m Q`.

        Incompressible, so the pressure is eliminated by projecting the force
        transverse to `k`. This is the one part of the model that is not
        local: activity anywhere moves fluid everywhere, which is why a bend
        on one side of the drop is felt on the other.
        """
        lib, fft, *_ = self._pack()
        kx, ky, k2s = (self._kx, self._ky, self._k2_safe) if self.device != "cpu" \
            else (self.kx, self.ky, self.k2_safe)
        mask = self._mask if self.device != "cpu" else self.mask
        sxx = -self.activity * mask * qxx
        sxy = -self.activity * mask * qxy
        fxx, fxy, fyy = fft.fft2(sxx), fft.fft2(sxy), fft.fft2(-sxx)
        fx = 1j * (kx * fxx + ky * fxy)
        fy = 1j * (kx * fxy + ky * fyy)
        projection = (kx * fx + ky * fy) / k2s
        ux = fft.ifft2((fx - kx * projection) / (self.viscosity * k2s)).real
        uy = fft.ifft2((fy - ky * projection) / (self.viscosity * k2s)).real
        return ux, uy

    def _grad(self, field):
        lib, fft, *_ = self._pack()
        kx, ky = (self._kx, self._ky) if self.device != "cpu" else (self.kx, self.ky)
        spectrum = fft.fft2(field)
        return fft.ifft2(1j * kx * spectrum).real, fft.ifft2(1j * ky * spectrum).real

    def _laplacian(self, field):
        lib, fft, *_ = self._pack()
        k2 = self._k2 if self.device != "cpu" else self.k2
        return fft.ifft2(-k2 * fft.fft2(field)).real

    # ------------------------------------------------------------------

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.step_index += 1
            if self.device != "cpu":
                qxx, qxy, mask = self._qxx, self._qxy, self._mask
            else:
                qxx, qxy, mask = self.qxx, self.qxy, self.mask

            ux, uy = self._flow(qxx, qxy)
            dx_ux, dy_ux = self._grad(ux)
            dx_uy, dy_uy = self._grad(uy)
            vorticity = 0.5 * (dy_ux - dx_uy)
            strain_xx = dx_ux
            strain_xy = 0.5 * (dy_ux + dx_uy)

            gxx_x, gxx_y = self._grad(qxx)
            gxy_x, gxy_y = self._grad(qxy)

            # The molecular field. `mask` is where the ordered well is: inside
            # the drop the equilibrium is |Q| = 1/sqrt(2), outside there is no
            # well and the alignment relaxes away.
            square = qxx * qxx + qxy * qxy
            h_xx = self.elasticity * self._laplacian(qxx) + self.landau * (mask - 2.0 * square) * qxx
            h_xy = self.elasticity * self._laplacian(qxy) + self.landau * (mask - 2.0 * square) * qxy

            # Advection, co-rotation with the vorticity, flow alignment with
            # the strain, and relaxation towards the molecular field.
            qxx = qxx + self.dt * (
                -(ux * gxx_x + uy * gxx_y) + 2.0 * vorticity * qxy
                + self.alignment * strain_xx + self.rotation * h_xx
            )
            qxy = qxy + self.dt * (
                -(ux * gxy_x + uy * gxy_y) - 2.0 * vorticity * qxx
                + self.alignment * strain_xy + self.rotation * h_xy
            )

            speed = (ux * ux + uy * uy) ** 0.5
            if self.device != "cpu":
                self._qxx, self._qxy, self._speed = qxx, qxy, speed
                self._wake = torch.maximum(self._wake * self.decay, self._defect_field(qxx, qxy))
            else:
                self.qxx, self.qxy, self.speed = qxx, qxy, speed
                self.wake = np.maximum(self.wake * self.decay, self._defect_field(qxx, qxy))

    # ------------------------------------------------------------------

    def _charge(self, qxx, qxy):
        """Topological charge on each plaquette, by the winding of the director.

        The director is a line, not an arrow: it comes back to itself after
        half a turn, so every angle difference is wrapped onto a half-turn
        before it is added up. Do that with a full turn and every defect
        reads as zero.
        """
        lib = torch if self.device != "cpu" else np
        angle = 0.5 * lib.arctan2(qxy, qxx) if lib is np else 0.5 * torch.atan2(qxy, qxx)
        roll = (lambda a, s, ax: np.roll(a, s, ax)) if lib is np else (lambda a, s, ax: torch.roll(a, s, ax))
        pi = math.pi

        def difference(first, second):
            return (second - first + pi / 2.0) % pi - pi / 2.0

        a = angle
        b = roll(angle, -1, 0)
        c = roll(roll(angle, -1, 0), -1, 1)
        d = roll(angle, -1, 1)
        winding = difference(a, b) + difference(b, c) + difference(c, d) + difference(d, a)
        return winding / (2.0 * pi)

    def _defect_field(self, qxx, qxy):
        """1 where a defect core sits, 0 elsewhere -- what the phosphor records."""
        charge = self._charge(qxx, qxy)
        if self.device != "cpu":
            return (charge.abs() > 0.2).to(charge.dtype) * self._mask
        return ((np.abs(charge) > 0.2).astype(np.float32)) * self.mask

    def defects(self) -> tuple[int, int]:
        """Counts of +1/2 and -1/2 defects inside the drop."""
        qxx, qxy = (self._qxx, self._qxy) if self.device != "cpu" else (self.qxx, self.qxy)
        charge = self._charge(qxx, qxy)
        if self.device != "cpu":
            charge = charge.cpu().numpy()
        inside = self.dish
        return int(((charge > 0.2) & inside).sum()), int(((charge < -0.2) & inside).sum())

    def order(self) -> float:
        """How aligned the drop still is as a whole, 1 at the start and ~0 in turbulence.

        Averaging Q over the drop and dividing by the average magnitude: a
        film pointing one way keeps its average, a torn one cancels itself.
        """
        qxx, qxy = self.arrays()[:2]
        inside = self.dish
        magnitude = float(np.hypot(qxx[inside], qxy[inside]).mean())
        vector = math.hypot(float(qxx[inside].mean()), float(qxy[inside].mean()))
        return vector / max(magnitude, 1e-9)

    def arrays(self) -> tuple[np.ndarray, ...]:
        if self.device != "cpu":
            return tuple(a.cpu().numpy() for a in (self._qxx, self._qxy, self._speed, self._wake))
        return self.qxx, self.qxy, self.speed, self.wake

    def state(self) -> tuple[np.ndarray, ...]:
        """One banked frame, compressed.

        Q and the wake are bounded -- by the Landau well and by construction --
        so eight bits is plenty, and the bloom and the tone curve wash the
        quantisation out long before anyone could see it. The speed is not
        bounded ahead of time and it is what the brightness is made of, so it
        is kept as float16 rather than given a reference the run has not
        finished measuring yet. Half a megabyte a state against two, which is
        the difference between banking the run in memory and not.
        """
        qxx, qxy, speed, wake = self.arrays()
        scale = 1.0 / (2.0 * 0.7071067811865476)
        return (
            np.clip((qxx * scale + 0.5) * 255.0, 0, 255).astype(np.uint8),
            np.clip((qxy * scale + 0.5) * 255.0, 0, 255).astype(np.uint8),
            speed.astype(np.float16),
            np.clip(wake * 255.0, 0, 255).astype(np.uint8),
        )

    @staticmethod
    def unpack(state: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        scale = 2.0 * 0.7071067811865476
        qxx = (state[0].astype(np.float32) / 255.0 - 0.5) * scale
        qxy = (state[1].astype(np.float32) / 255.0 - 0.5) * scale
        return qxx, qxy, state[2].astype(np.float32), state[3].astype(np.float32) / 255.0

    def metric(self) -> float:
        """Not used -- the schedule is built from the banked run, not read live."""
        return float(sum(self.defects()))

    # ------------------------------------------------------------------

    def samples(
        self,
        state: tuple[np.ndarray, ...],
        scale: float,
        centre: tuple[float, float],
        seeds: int = 320_000,
        walk: int = 9,
        stride: float = 0.55,
        generator: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Short streamlines along the director: the film's own brush strokes.

        A director field drawn as a value per pixel is a smooth wash with no
        filaments in it, and filaments are what the subject is made of. Walking
        a few steps along the alignment from a scatter of seeds puts the
        strokes back, and because the walk follows the field, the strokes bunch
        where the film bends -- which is where the tearing happens.

        Returns points in frame coordinates, a value per point, and a weight
        per point. Q is interpolated, not the angle: an angle cannot be
        averaged across the wrap and every stroke crossing it would kink.
        """
        qxx, qxy, speed, wake = self.unpack(state)
        size = self.size
        generator = np.random.default_rng(4) if generator is None else generator

        # Seeds inside the drop only. The black margin is black because there
        # is no film there, not because the drawing stops.
        angle = generator.uniform(0.0, 2.0 * math.pi, seeds)
        span = self.radius * np.sqrt(generator.uniform(0.0, 1.0, seeds))
        x = (size - 1) / 2.0 + span * np.cos(angle)
        y = (size - 1) / 2.0 + span * np.sin(angle)

        def bilinear(field, px, py):
            x0 = np.floor(px).astype(np.int32)
            y0 = np.floor(py).astype(np.int32)
            fx, fy = px - x0, py - y0
            x0 %= size
            y0 %= size
            x1, y1 = (x0 + 1) % size, (y0 + 1) % size
            return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy)
                    + field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)

        points = np.empty((walk * seeds, 2), dtype=np.float32)
        values = np.empty(walk * seeds, dtype=np.float32)
        weights = np.empty(walk * seeds, dtype=np.float32)
        inside = np.empty(walk * seeds, dtype=bool)
        previous_x = np.zeros(seeds)
        previous_y = np.zeros(seeds)
        for index in range(walk):
            axx = bilinear(qxx, x, y)
            axy = bilinear(qxy, x, y)
            heading = 0.5 * np.arctan2(axy, axx)
            step_x, step_y = np.cos(heading), np.sin(heading)
            if index:
                # The director has no sign, so `arctan2` flips arbitrarily from
                # one cell to the next. Keep walking the way we were walking.
                flipped = (step_x * previous_x + step_y * previous_y) < 0.0
                step_x = np.where(flipped, -step_x, step_x)
                step_y = np.where(flipped, -step_y, step_y)
            else:
                sign = np.where(generator.random(seeds) < 0.5, -1.0, 1.0)
                step_x, step_y = step_x * sign, step_y * sign
            previous_x, previous_y = step_x, step_y

            block = slice(index * seeds, (index + 1) * seeds)
            points[block, 0] = centre[0] + (x - (size - 1) / 2.0) * scale
            points[block, 1] = centre[1] + (y - (size - 1) / 2.0) * scale
            values[block] = bilinear(wake, x, y)
            weights[block] = bilinear(speed, x, y)
            # A stroke started inside can walk out, and nine of them abreast draw
            # a fringe of hair round the drop -- a structure the model never made
            # and the one thing in the frame that looks like a bug. Seeding
            # further in would thin the rim instead; dropping the samples that
            # left keeps the density even right up to the edge.
            inside[block] = np.hypot(x - (size - 1) / 2.0, y - (size - 1) / 2.0) <= self.radius
            x = x + stride * step_x
            y = y + stride * step_y
        return points[inside], values[inside], weights[inside]

