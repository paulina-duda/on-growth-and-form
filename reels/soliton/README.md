# Soliton

**Artificial Life** · Lenia, and what one collision does

<p align="center"><img src="loop.webp" width="320"></p>

> *Every one of these was stable on its own.*

Conway's Life counts eight neighbours, compares the count to two integers, and
switches a cell on or off. **Lenia** keeps the shape of that idea and makes
every part of it continuous: a cell holds a real number instead of a bit, the
neighbourhood becomes a smooth ring instead of a square of eight, the birth and
survival intervals become one smooth growth curve, and time is divided so a step
moves the field by a tenth of the growth instead of all of it. Four changes,
none of them clever.

What comes out are not blinkers and gliders but **solitons** — lumps of
continuous field, smooth-edged and internally structured, that hold themselves
together while they travel. The one here is a ring with a core, about sixty
cells across, swimming at a third of a cell per step.

Nothing in it is designed. A creature in Lenia is a point in a space of
parameters and initial conditions, and almost all of that space is lethal in one
of two ways: the field collapses to nothing, or it grows without limit until it
fills the world. Everything interesting lives in the narrow band between, and it
has to be searched for.

Then twelve copies of one creature are put in a world that wraps in both
directions, and the piece is what they do to each other. Stability against the
empty field is not stability against a neighbour.

```
Lenia · continuous automaton (Chan 2019)
one ring kernel · one growth curve
twelve creatures, found by search
```

<sub>Lenia — Bert Wang-Chak Chan, 2019, cited in frame. 1080 × 1920, 10 s — one
of two pieces on the account that needed longer than eight.
[On Instagram →](https://instagram.com/ekspertodniczego)</sub>
