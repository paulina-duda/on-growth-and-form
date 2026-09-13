# Source

The code that produced two of the clips on the front page, each one runnable on
its own. Both were checked against the published cut frame by frame: the still
they write is **pixel-identical** to the one that went out, zero pixels
differing.

| Piece | Edition | What it is |
| --- | --- | --- |
| [`phyllotaxis/`](phyllotaxis/) | Wetware | inhibition-field organ placement — a simulation |
| [`hydrocreatures/`](hydrocreatures/) | Biomorph | one closed parametric curve — no simulation at all |

The pair is deliberate. One is a process that computes its own form step by
step and could not be written down in closed form; the other *is* the closed
form, and only looks alive. Putting them side by side is the argument the whole
account is built on.

## Running them

```bash
pip install numpy pillow          # plus an ffmpeg built with libx264
cd phyllotaxis && python3 phyllotaxis.py --preview     # one still
cd phyllotaxis && python3 phyllotaxis.py               # the 8 s clip
```

```bash
cd hydrocreatures && python3 hydrocreatures.py --variant neon --no-caption --stills 0
cd hydrocreatures && python3 hydrocreatures.py --variant neon --no-caption
```

Output lands in `../out/`. `--variant` picks between `aurora`, `reef` and
`neon`; the published cut is `neon` with `--no-caption`, which drops the data
block and re-centres the animals.

**ffmpeg matters.** conda-forge's default build is the LGPL one and carries only
libopenh264, which advertises H.264 and then fails at runtime. Ask for
`ffmpeg=*=gpl*` by name, or use the system one. Both scripts probe for a working
encoder before they start.

## What is here and what is not

`glow.py` is the shared drawing module — palette building, additive splatting,
log-density tone mapping, multi-scale bloom, and the caption typography. It
appears once per piece because the two editions carry slightly different copies
of it in the working repository, and publishing them merged would mean
publishing something neither clip was rendered with.

`phyllotaxis.py` is an extraction: the model class and the frame pipeline as
shipped, with the edition registry and the other pieces removed.
`hydrocreatures.py` is the working script verbatim, with only its font and
output paths repointed at this repository.

The parameters in both are the ones the published clips were run at. Nothing
was tidied for display — the comments explaining why a number is what it is are
the ones that were written when it cost something to find out.
