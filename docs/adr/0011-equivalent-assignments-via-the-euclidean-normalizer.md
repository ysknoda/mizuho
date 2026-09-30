# Equivalent Wyckoff assignments are merged before ranking, via the Euclidean normalizer

Two Wyckoff assignments can describe the same crystal. For spinel, Phase 1's top candidate
`Mg@b Al@c O@e` and the published `Mg@a Al@d O@e` are one structure: in Fd-3m (origin choice 1,
mizuho's setting) the origin shift (½, ½, ½) maps the space group onto itself and exchanges 8a with
8b and 16c with 16d. Both scored 0.000000. The two anatase CIFs supplied for verification land in
mizuho's I4₁/amd setting as `Ti@a O@e` and `Ti@b O@e` — the same mineral, two representatives.

The operations that do this are the space group's **Euclidean normalizer**: the affine maps N
with N G N⁻¹ = G that preserve the metric. Its discrete part (origin shifts, extra point
operations) makes duplicate assignments. Its continuous part is the **floating origin** of a polar
space group — 68 of the 230 groups — where translating the whole structure along the polar axis
changes nothing, so one free coordinate is redundant. These are one mathematical object and are
handled by one mechanism.

## Decision

- **Derive the normalizer from mizuho's own symmetry-operation tables**, by testing N G N⁻¹ = G
  directly over candidate origin shifts (combinations of 0, 1/8, 1/4, 1/2 along each axis) and the
  metric-preserving point operations. The continuous part falls out of the same test as the polar
  directions. This is the rule mizuho's Wyckoff database already follows — derived from spglib's
  operations, no external table — and transcribing International Tables Vol. A would break it.
- **Use the normalizer for the general metric of the crystal family.** A normalizer depends on the
  metric: an orthorhombic group with a = b gains an a↔b swap. mizuho bounds each axis separately in
  `lattice_bounds`, so a structure with a and b exchanged is a different point of the search, not
  a duplicate.
- **Merge after enumeration, before Phase 1 ranking.** Each enumerated assignment is mapped to a
  canonical representative of its equivalence class, and duplicates are dropped. Ranking is where
  Phase 1 spends its time (adamite 135-172 s, topaz 601-1152 s); enumeration is cheap. Merging
  here keeps the `dfs` / `best_first` oracle relationship intact, which pruning during enumeration
  would not.
- **The rank gate compares equivalence classes, not letters.** Otherwise a canonical
  representative such as spinel's `Mg@b Al@c O@e` would make the published topology look absent.
- Whether to remove the continuous (floating-origin) redundancy from the optimiser is decided by
  measurement on wurtzite, with and without pinning; the normalizer computes it either way.

## Consequences

- Validation needs no external table: the defining property N G N⁻¹ = G; the spinel case above;
  and for every merged pair, spglib confirming that the two built structures are the same crystal.
- The expected reduction in candidates is up to the normalizer's index over the group (8 for many
  primitive orthorhombic groups — Pnma's normalizer is Pmmm with halved axes). This is an
  expectation, confidence about 85%, and is measured when implemented.
- Terminology: these are **equivalent assignments**. `CONTEXT.md` already reserves *orbit* for the
  set of symmetry-equivalent points of one Wyckoff position.
