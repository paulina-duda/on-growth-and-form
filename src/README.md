# Source

The code that produced every clip on the front page. Each edition keeps its own
renderer, because that is how the working repository is actually laid out — the
four carry slightly different copies of the drawing module, and merging them
would mean publishing something none of the clips were rendered with.

| Directory | Pieces | Run |
| --- | --- | --- |
| [`wetware/`](wetware/) | Phyllotaxis, Stripe, Tear | `render.py --edition phyllotaxis --duration 8` |
| [`substrate/`](substrate/) | Defect | `render.py --edition defect` |
| [`alife/`](alife/) | Soliton | `render.py --edition soliton --duration 10` |
| [`biomorph/`](biomorph/) | Hydrocreatures | `hydrocreatures.py --variant neon --no-caption` |

## Verified, not asserted

Every one of the six was run and its cover frame compared against the published
cut pixel by pixel:

| Piece | Command | Result |
| --- | --- | --- |
| Phyllotaxis | `render.py --edition phyllotaxis --duration 8` | identical, max difference 0 |
| Stripe | `render.py --edition stripe --duration 8` | identical, max difference 0 |
| Tear | `render.py --edition tear3 --palette prism --duration 10` | identical, max difference 0 |
| Defect | `render.py --edition defect` | identical, max difference 0 |
| Soliton | `render.py --edition soliton --duration 10` | identical, max difference 0 |
| Hydrocreatures | `hydrocreatures.py --variant neon --no-caption --stills 0` | identical, max difference 0 |

Getting there found three real discrepancies that reading the code would not
have shown: the settle is `steps_per_frame * (frames - 1)` and not one step
more, frame zero is the cover with the run starting over behind it, and
`--cell-reference` is the 92nd percentile rather than the 99th.

## Running them

```bash
conda env create -f ../environment.yml
conda activate on-growth-and-form
cd wetware && python3 render.py --edition phyllotaxis --duration 8 --preview
```

`--preview` writes the cover still and stops, which takes about a second;
without it you get the mp4 and its cover in `../out/`.

**ffmpeg matters.** conda-forge's default build is the LGPL one and carries only
libopenh264, which advertises H.264 and then fails at runtime. `environment.yml`
asks for the GPL build by name. Every renderer probes for a working encoder
before it starts rather than discovering the problem three minutes into a clip.

## What was changed on the way here

These are trimmed copies of the working renderers, not rewrites. Each had its
edition registry cut down to the pieces shown on the front page, and with it the
palettes, model classes, timeline functions and command-line flags belonging to
everything else. `build()` in the wetware renderer was rewritten to cover only
the three kinds it still serves. Output and font paths were repointed here, and
the wetware renderer gained a `--preview` flag it did not have.

Cross-references in the comments were rewritten where they named a piece that is
not published here. The lesson each one records is kept; the name is not.

Nothing else was touched. The parameters are the ones the clips were run at, and
the comments explaining why a number is what it is are the ones written when it
cost something to find out — including the ones that record a mistake.

`biomorph/hydrocreatures.py` is the working script verbatim, paths aside.

## Credit

`biomorph/hydrocreatures.py` is inspired by the generative sketches of
**@yuruyurau**, who posts them on X: [x.com/yuruyurau](https://x.com/yuruyurau).
The implementation, the harmonic search, the composition and the rendering here
are my own — no code of his was used, and he is not an author of this.

## Licensing

Code here is under [PolyForm Noncommercial 1.0.0](../LICENSE): free for any
noncommercial purpose, **commercial use requires separate written permission.**

The clips and stills this code produces are **not** covered by that licence —
they are © 2026 Paulina Duda, all rights reserved.

`fonts/` is IBM Plex Mono under the SIL Open Font License 1.1; see
[`fonts/NOTICE.md`](fonts/NOTICE.md).
