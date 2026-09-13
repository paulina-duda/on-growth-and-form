# Stripe

**Wetware** · a zebrafish pattern the tissue argues out

<img src="loop.webp" width="320" align="right">

> *Turing predicted chemicals. These are cells.*

A zebrafish stripe is not a chemical pattern. It is an argument between two
kinds of cell.

Turing showed in 1952 that two substances diffusing at different rates are
enough to make stripes, and for a long time the zebrafish was the textbook case.
Then the cells were killed off one at a time with a laser, and what grew back
gave the interaction away. The shape Turing needed was there — a cell supports
its own kind close in and suppresses it further out — but **the morphogens were
not chemicals. They were whole cells**: the short- and long-range terms are
carried by the pigment cells themselves — black melanophores and yellow
xanthophores — each responding to how many of the other kind are nearby.

**The binary flip is the model's, not the fish's.** A site here holds one of two
states and switches, stochastically, once per step. That stands in for a much
slower biological story: real tissue rearranges through cell death, division,
migration and the differentiation of precursors, and a melanophore does not
simply turn into a xanthophore. What survives the abstraction is the *shape* of
the interaction, which is the part the pattern depends on.

```
drive = G(σ_near) * f  −  w · G(σ_far) * f        f = +1 black, −1 yellow
a cell becomes sign(drive), with probability λ per step
```

So the pattern is computed by the tissue rather than painted onto it. Every
grain here is one animal cell that decided, and can still change its mind. In
eight seconds, 667,000 of them do.

The fish also never stops growing, and that is the rest of it. The two ranges
are fixed — they are the reach of one cell's processes, and a cell does not get
longer because the fish does. A stripe therefore has a width it wants, and skin
that keeps widening carries the existing stripes apart until the gap between
them is wide enough to hold another one, at which point a new stripe nucleates
in the middle of it.

**Nothing counts them.** The number of stripes is the height of the skin divided
by a width that no cell chose.

Colour is which of the two cells won that patch of skin. The bright edges are
the ones that have just changed their minds.

```
zebrafish pigment pattern  ·  Nakamasu 2009
support close · suppress far · switch
33,000 cells become 118,000 · 667,000 switches
the stripe's width is fixed; the skin's is not
```

<sub>After Nakamasu et al. 2009, cited in frame. 1080 × 1920, 8 s.
[On Instagram →](https://instagram.com/ekspertodniczego)</sub>
