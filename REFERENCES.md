# References

The models behind the pieces on the front page. Every citation here was checked
against Crossref rather than typed from memory — author list, year, journal,
volume and pages all resolved from the DOI.

Where a piece rests on a published model, the citation is also burned into its
frame, so it travels with the clip rather than living only in a caption.

---

### Phyllotaxis

**Douady, S. & Couder, Y. (1992).** Phyllotaxis as a physical self-organized
growth process. *Physical Review Letters* **68**(13), 2098–2101.
[10.1103/PhysRevLett.68.2098](https://doi.org/10.1103/PhysRevLett.68.2098)

The rule the piece runs: each primordium is placed where those already down
inhibit least, on a rim of fixed size, with the tissue underneath carrying
everything outwards. The same authors got the pattern out of ferrofluid drops
repelling each other in a dish of oil, with no biology in it at all — which is
the useful thing to remember about it.

→ [`reels/phyllotaxis/`](reels/phyllotaxis/) · [`src/wetware/`](src/wetware/)

### Stripe

**Nakamasu, A., Takahashi, G., Kanbe, A. & Kondo, S. (2009).** Interactions
between zebrafish pigment cells responsible for the generation of Turing
patterns. *Proceedings of the National Academy of Sciences* **106**(21),
8429–8434.
[10.1073/pnas.0808622106](https://doi.org/10.1073/pnas.0808622106)

The paper that moved the zebrafish stripe off two diffusing chemicals and onto
the pigment cells themselves, by ablating cells one at a time with a laser and
watching what grew back. The interaction keeps Turing's shape — support close
in, suppress further out — with cells standing where the morphogens were.

→ [`reels/stripe/`](reels/stripe/) · [`src/wetware/`](src/wetware/)

### Tear

**Ferrante, E., Turgut, A. E., Dorigo, M. & Huepe, C. (2013).** Elasticity-based
mechanism for the collective motion of self-propelled particles with spring-like
interactions: a model system for natural and artificial swarms. *Physical Review
Letters* **111**(26), article 268302.
[10.1103/PhysRevLett.111.268302](https://doi.org/10.1103/PhysRevLett.111.268302)

**Prakash, V. N., Bull, M. S. & Prakash, M. (2021).** Motility-induced fracture
reveals a ductile-to-brittle crossover in a simple animal's epithelia. *Nature
Physics* **17**(4), 504–511.
[10.1038/s41567-020-01134-7](https://doi.org/10.1038/s41567-020-01134-7)

Two papers, and the piece is the join between them. Ferrante and colleagues give
the alignment rule — a particle turns toward the force its neighbours put on it,
which is enough to line thousands of them up with nothing coordinating them.
Prakash and colleagues filmed placozoan tissue tearing under its own crawling.
The bonds that yield are this implementation's, not theirs.

→ [`reels/tear/`](reels/tear/) · [`src/wetware/`](src/wetware/)

### Defect

**Sanchez, T., Chen, D. T. N., DeCamp, S. J., Heymann, M. & Dogic, Z. (2012).**
Spontaneous motion in hierarchically assembled active matter. *Nature* **491**
(7424), 431–434.
[10.1038/nature11591](https://doi.org/10.1038/nature11591)

Microtubules, kinesin and ATP, and nothing alive. The alignment is modelled
Beris–Edwards with one elastic constant, coupled to Stokes flow, with an active
stress proportional to the alignment itself — the term that turns alignment into
flow and lets the flow bend the alignment back.

→ [`reels/defect/`](reels/defect/) · [`src/substrate/`](src/substrate/)

### Soliton

**Chan, B. W.-C. (2019).** Lenia: biology of artificial life. *Complex Systems*
**28**(3), 251–286.
[10.25088/ComplexSystems.28.3.251](https://doi.org/10.25088/ComplexSystems.28.3.251)
· preprint: [arXiv:1812.05433](https://arxiv.org/abs/1812.05433)

Conway's Life with four things made continuous: a real-valued cell, a smooth
ring kernel, one smooth growth curve in place of birth and survival integers,
and a timestep that moves the field by a fraction of the growth. What comes out
are solitons — lumps of field that hold themselves together while they travel.

→ [`reels/soliton/`](reels/soliton/) · [`src/alife/`](src/alife/)

---

### Hydrocreatures

No paper. The piece is a closed-form parametric curve and a search over
harmonics, and both are mine. It is **inspired by the generative sketches of
@yuruyurau**, who posts them on X at
[x.com/yuruyurau](https://x.com/yuruyurau) — no code of his was used, and he is
not an author of it.

→ [`reels/hydrocreatures/`](reels/hydrocreatures/) · [`src/biomorph/`](src/biomorph/)

---

### On the project's name

**Thompson, D'A. W. (1917).** *On Growth and Form.* Cambridge University Press.

The argument that the shapes living things take are set by physics and
mathematics, not by descent alone — made with geometry, by someone who had no
way to run any of it. These renders are the cheap test of that claim: write the
rule, let it run, and see whether the animal falls out.
