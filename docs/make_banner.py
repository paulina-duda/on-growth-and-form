#!/usr/bin/env python3
"""Draw `banner.svg` -- the wordmark at the top of the README.

It was ASCII art in a fenced code block until it met a phone. A monospace block
font needs four columns per letter to stay legible, so sixteen letters come to
79 columns, and GitHub renders that at 676 px whatever the screen is. The README
column on a 375 px phone is **309 px**, so the wordmark was cut off mid-word for
anyone arriving from Instagram -- which is most people.

GitHub strips CSS from READMEs, so there is no media query to escape with. An
image is the way out: GitHub applies `max-width: 100%` to images, so an SVG
shows at its natural size on a desktop and scales down to fit a phone, and the
letterforms stay crisp at both because they are vectors rather than pixels.

The plate is black with white letters rather than theme-coloured, for two
reasons. It reads identically in GitHub's light and dark themes without needing
two files and a `<picture>` switch, and black-with-white-type is what every
frame in this project looks like anyway.

The tagline is drawn from the same glyphs at a smaller unit rather than set as
`<text>`. An SVG loaded through `<img>` gets no web fonts and no promise about
how `letter-spacing` is measured, and the first version of this file proved it
by running a `<text>` tagline straight off both edges of the plate. Blocks have
a width that can be calculated, so the line is centred and fits by construction.

    python3 make_banner.py

Needs nothing but the standard library.
"""

from __future__ import annotations

from pathlib import Path

WORD = "EKSPERTODNICZEGO"
TAGLINE = "BIOLOGY IS THE ORIGINAL ALGORITHM"

# Four columns by six rows per letter. Narrower than this stops being a font
# and starts being noise; wider does not fit a phone even as an image.
GLYPHS = {
    "E": ["1111", "1000", "1110", "1000", "1000", "1111"],
    "K": ["1001", "1010", "1100", "1100", "1010", "1001"],
    "S": ["1111", "1000", "1111", "0001", "0001", "1111"],
    "P": ["1111", "1001", "1111", "1000", "1000", "1000"],
    "R": ["1111", "1001", "1111", "1100", "1010", "1001"],
    "T": ["1111", "0110", "0110", "0110", "0110", "0110"],
    "O": ["1111", "1001", "1001", "1001", "1001", "1111"],
    "D": ["1110", "1001", "1001", "1001", "1001", "1110"],
    "N": ["1001", "1101", "1101", "1011", "1011", "1001"],
    "I": ["1111", "0110", "0110", "0110", "0110", "1111"],
    "C": ["1111", "1000", "1000", "1000", "1000", "1111"],
    "Z": ["1111", "0011", "0110", "1100", "1000", "1111"],
    "G": ["1111", "1000", "1011", "1001", "1001", "1111"],
    "B": ["1110", "1001", "1110", "1001", "1001", "1110"],
    "L": ["1000", "1000", "1000", "1000", "1000", "1111"],
    "Y": ["1001", "1001", "0110", "0110", "0110", "0110"],
    "H": ["1001", "1001", "1111", "1001", "1001", "1001"],
    "A": ["0110", "1001", "1001", "1111", "1001", "1001"],
    "M": ["1001", "1111", "1111", "1001", "1001", "1001"],
    " ": ["0000", "0000", "0000", "0000", "0000", "0000"],
}

UNIT = 10          # one glyph pixel of the wordmark, in SVG user units
GAP = 1            # blank columns between letters
PAD_X, PAD_TOP = 46, 40
RULE_GAP, TAG_GAP, PAD_BOTTOM = 26, 22, 32

# The tagline is a caption, not a second wordmark, so it runs at roughly a third
# of the wordmark's glyph size -- close to the ratio the old ASCII lockup had,
# where the tagline was single characters and the wordmark was built out of
# many. At this unit the line comes to 574 units against the wordmark's 790, so
# it sits comfortably inside and needs no measuring at render time.
TAG_UNIT = 3.5

INK = "#ECECEC"
PLATE = "#08080A"
RULE = "#4A4A4E"
TAG = "#A0A0A6"


def main() -> None:
    cols = len(WORD) * (4 + GAP) - GAP
    mark_w, mark_h = cols * UNIT, 6 * UNIT
    width = mark_w + 2 * PAD_X
    rule_y = PAD_TOP + mark_h + RULE_GAP
    tag_y = rule_y + 2 + TAG_GAP
    height = tag_y + 6 * TAG_UNIT + PAD_BOTTOM

    def draw(text: str, left: float, top: float, unit: float) -> list[str]:
        out = []
        for index, letter in enumerate(text):
            origin = left + index * (4 + GAP) * unit
            for row, bits in enumerate(GLYPHS[letter]):
                run = 0
                # Merge horizontal runs into one rect each: fewer nodes, and no
                # hairline seams between neighbouring squares when the browser
                # scales the whole thing down to a phone.
                for col in range(5):
                    if col < 4 and bits[col] == "1":
                        run += 1
                        continue
                    if run:
                        x = origin + (col - run) * unit
                        out.append(
                            f'<rect x="{x:g}" y="{top + row * unit:g}" '
                            f'width="{run * unit:g}" height="{unit:g}"/>'
                        )
                        run = 0
        return out

    tag_w = (len(TAGLINE) * (4 + GAP) - GAP) * TAG_UNIT
    letters = "\n    ".join(draw(WORD, PAD_X, PAD_TOP, UNIT))
    tag = "\n    ".join(draw(TAGLINE, (width - tag_w) / 2, tag_y, TAG_UNIT))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" \
width="{width}" height="{height}" role="img" \
aria-label="{WORD} — biology is the original algorithm">
  <rect width="{width}" height="{height}" fill="{PLATE}"/>
  <g fill="{INK}" shape-rendering="crispEdges">
    {letters}
  </g>
  <rect x="{PAD_X}" y="{rule_y}" width="{mark_w}" height="2" fill="{RULE}"/>
  <g fill="{TAG}" shape-rendering="crispEdges">
    {tag}
  </g>
</svg>
"""
    destination = Path(__file__).resolve().parent / "banner.svg"
    destination.write_text(svg)
    print(f"Saved {destination}  ({width:g} x {height:g}), tagline {tag_w:g} wide")


if __name__ == "__main__":
    main()
