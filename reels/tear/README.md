# Tear

**Wetware** · an animal that pulls itself apart

<img src="loop.webp" width="320" align="right">

> *It reproduces by disagreeing with itself.*

*Trichoplax adhaerens* is about as little animal as an animal gets: two sheets
of epithelium with a layer of fibre cells between them, no nerves, no muscle, no
organs, no front and no back. It crawls on a carpet of cilia, and **every
ventral cell walks on its own.** Nothing tells any of them which way the animal
is going.

Cells are tied to their neighbours, so a cell that walks gets pulled by the ones
around it, and it turns toward the pull:

```
ẋᵢ = v₀·n̂ᵢ + μ·Fᵢ           θ̇ᵢ = β·(Fᵢ · n̂ᵢ⊥) + ηᵢ
```

That single rule is enough to line thousands of cells up into one heading
(Ferrante et al. 2013). But nothing coordinates them globally, so a large enough
animal ends up holding patches that agree internally and disagree with each
other — and the tissue between those patches stretches.

Bonds past a strain threshold give way. Bonds re-form between nearby cells of
the same animal, so the sheet **yields instead of shattering**, and an animal
stretched far enough comes apart in two. That is how this animal reproduces, and
Prakash, Bull & Prakash filmed it in 2021: motility-induced fracture, a body
torn by its own crawling.

**In this model, growth acts as the reset clock.** Cells divide in place, so
each half grows back to the size at which it tears again, and the piece keeps
producing events instead of settling. That is an implementation choice, made
because without it the run is a one-off relaxation: measured before building,
sixteen animals shattered into thirty-seven pieces in the first quarter and
nothing tore afterwards — 85.2% of the change, then 0.0%. It is not a claim
about how often a real *Trichoplax* divides.

Colour is which animal a cell belongs to, so a tear is one colour becoming two.
How stretched a cell is rides in the brightness instead: a line the tissue is
about to open along lights up before it opens.

```
Trichoplax adhaerens  ·  fission
walk · turn to the pull · stretch · give
motility-induced fracture, Prakash 2021
```

<sub>Ferrante et al. 2013 for the coupling, Prakash et al. 2021 for the
fracture; both cited in frame. 1080 × 1920, 10 s.
[On Instagram →](https://instagram.com/ekspertodniczego)</sub>
