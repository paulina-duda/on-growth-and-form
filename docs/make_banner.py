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

Only the wordmark and its rule are drawn here. The tagline stays in the README
as real text: an SVG loaded through `<img>` gets no web fonts and no guarantee
about how `letter-spacing` is measured, and the first version of this file
proved it by running the tagline straight off both edges of the plate. Text
that has to be read belongs in the markup.

    python3 make_banner.py

Needs nothing but the standard library.
"""

from __future__ import annotations

from pathlib import Path

WORD = "EKSPERTODNICZEGO"

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
}

UNIT = 10          # one glyph pixel, in SVG user units
GAP = 1            # blank columns between letters
PAD_X, PAD_TOP = 46, 40
RULE_GAP, PAD_BOTTOM = 26, 26

INK = "#ECECEC"
PLATE = "#08080A"
RULE = "#4A4A4E"


def main() -> None:
    cols = len(WORD) * (4 + GAP) - GAP
    mark_w, mark_h = cols * UNIT, 6 * UNIT
    width = mark_w + 2 * PAD_X
    rule_y = PAD_TOP + mark_h + RULE_GAP
    height = rule_y + 2 + PAD_BOTTOM

    rects = []
    for index, letter in enumerate(WORD):
        origin = PAD_X + index * (4 + GAP) * UNIT
        for row, bits in enumerate(GLYPHS[letter]):
            run = 0
            # Merge horizontal runs into one rect each: fewer nodes, and no
            # hairline seams between neighbouring squares when the browser
            # scales the whole thing down to a phone.
            for col in range(5):
                on = col < 4 and bits[col] == "1"
                if on:
                    run += 1
                    continue
                if run:
                    x = origin + (col - run) * UNIT
                    rects.append(
                        f'<rect x="{x}" y="{PAD_TOP + row * UNIT}" '
                        f'width="{run * UNIT}" height="{UNIT}"/>'
                    )
                    run = 0

    letters = "\n    ".join(rects)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" \
width="{width}" height="{height}" role="img" aria-label="{WORD}">
  <rect width="{width}" height="{height}" fill="{PLATE}"/>
  <g fill="{INK}" shape-rendering="crispEdges">
    {letters}
  </g>
  <rect x="{PAD_X}" y="{rule_y}" width="{mark_w}" height="2" fill="{RULE}"/>
</svg>
"""
    destination = Path(__file__).resolve().parent / "banner.svg"
    destination.write_text(svg)
    print(f"Saved {destination}  ({width} x {height})")


if __name__ == "__main__":
    main()
