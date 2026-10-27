# A composition-free Wyckoff search: compositions come out of the tree, not into it

Every search so far took a composition as input; Phase 1 enumerated assignments whose atom
counts are a whole multiple of it. For an Al-O-N search the composition is the unknown. The
legacy `auto_composition` mode listed charge-neutral compositions per space group from the
Wyckoff multiplicities and ran a composition search for each, but it ran Phase 0's PUCT screen
(ADR-0021: does not reach known answers) instead of the exhaustive strategy, had no record or
resume, no supergroup promotion (ADR-0010), no symmetry-first ranking (ADR-0026), and kept only
the 30 smallest compositions per group without saying so. Design review of 2026-10-26.

## Decision

- **Input.** Omit `composition`; state `elements: [Al, O, N]`, their `ionic_charges` (all
  non-zero, at least one cation and one anion) and `tree_search.max_atoms`. Everything else is
  as for a fixed composition, including `mode: auto_spacegroup` with the exhaustive strategy or
  a fixed `spacegroup`.
- **The tree (Q1 (a)).** [space group] -> [element on a Wyckoff letter, repeatedly] -> a node is
  accepted when it is charge-neutral within the tolerance and holds every element; it is
  expanded further either way, since a neutral cell plus more sites can be neutral again. The
  composition is the accepted leaf's reduced formula (`search.base.reduced_formula`, elements in
  the stated order, counts of 1 omitted). Pruning is the existing charge-reachability bound and
  the atom bound; the composition-ratio checks do not apply. The enumeration is unbudgeted.
- **The cell bound is the primitive cell's atom count (Q2, Q7).** `tree_search.max_atoms`
  bounds the primitive cell; a Wyckoff multiplicity counts the conventional cell, so the bound
  is scaled by the group's centring factor (P 1, A/B/C/I 2, R 3, F 4;
  `symmetry.centring_factor`). Z is an output: Al3O3N at max_atoms 21 reaches Z <= 3, Al5O6N
  Z = 1. One bound for every composition, because it is the DFT cost; a bound on Z instead
  would let large formulas grow without limit. `max_formula_units` and `max_atoms_per_fu` are
  rejected in this mode.
- **Every element is required (Q3)** unless `tree_search.require_all_elements: false`.
- **Two sites on one fixed point are excluded.** A zero-free-parameter letter is used at most
  once per assignment (the same-letter exclusion of the zero-charge mode, applied here always,
  and enforced again after letter expansion): two orbits on the same fixed points are two atoms
  on top of each other. The ionic fixed-composition mode keeps its behaviour (records show it
  optimising such assignments, e.g. TiO2 `Ti@a + Ti@a`, error 2001).
- **Phase 2 cap per (space group, composition) (Q4).** Phase 1 groups the accepted assignments
  by composition; each group gets the config a fixed-composition run of that composition would
  get (`build_sg_config`, so ADR-0028's derivation and the density term see a composition),
  is merged under the Euclidean normalizer and ranked by geometric feasibility, and keeps
  `tree_search.max_candidates` (default 50) for Phase 2
  (`pipeline.search_and_rank_compositions`). A cap per space group would let the cheap surrogate hand
  the slots to the smallest compositions.
- **Each composition's config states its largest Z.** geometry's letter-feasibility table keeps
  the letters whose multiplicity fits composition x `max_formula_units`, and the staged ranking
  scores a letter missing from it as perfect (`letter_feasibility.get(..., 0.0)`). A
  composition's config has no `max_formula_units` of its own, so it got the default 4: at 21
  atoms per primitive cell F and R groups (225, 227, 166, 167) lost 5-15 letters their own
  assignments use. `pipeline.max_formula_units` now gives each composition the largest Z
  among its assignments, which covers every used letter and no letter beyond them.
- **One ranking per composition (Q5).** `run` prints one ranking per composition (ADR-0026's
  order inside each) and a composition summary; `--top N` writes N classes per composition.
  The record carries each entry's composition and resumes as before.
  `scripts/export_classes.py` keeps classes apart per composition, one subdirectory each, with
  a `composition` column in `classes.csv`. `--best N` also exports each composition's N
  lowest-error classes whatever their error (a `within_threshold` column marks them), so a
  composition that never reaches zero error still gets its closest structures; run on a record
  still being written, it reports a run's progress (`runs/alon/progress/<MMDD_HHMM>/`). Comparing errors across compositions is allowed
  (every term is normalised, ADR-0009) but means nothing among zero-error entries.
- **`auto_composition` is removed (Q6),** with `discover_compositions`; a config that still
  sets `auto_composition` or `auto_composition_settings` is rejected with a pointer here (as
  ADR-0009 did for `normalize_penalties`). `composition_pool` is unchanged.
- **Per-anion coordination numbers.** With several anions, ADR-0028's mean m fixes only their
  combined count. `anion_coordination_numbers: {O: [2, 4], N: [3, 4]}` states how it splits,
  in place of [floor(m), ceil(m)], at the derived cutoff.

## Verified

- **Oracle 1, brute force.** In 221, 229, 225, 167, 186, 12, 15 and 2 (P, I, F, R and C lattices,
  cubic to triclinic), Al-O-N at 7-14 atoms and 4-5 sites: both search strategies return exactly
  the multisets an exhaustive product over every (element, letter) yields under the same rules
  (`tests/unit/test_composition_free_search.py`).
- **Oracle 2, Q6 (i).** In the same groups the result equals, composition by composition, the
  fixed-composition search with the same conventional-cell bound, minus its fixed-point
  overlaps.
- **Known answer, Q6 (ii).** Elements Mg, Al, O, cited CN Mg 4 and Al 6, MgAl2O4's density,
  Fd-3m, 14 atoms per primitive cell, 3 sites: the only composition found is MgAl2O4 (4
  assignments, 2 after the normalizer merge); the published Mg@a Al@d O@e ranks first and
  optimises to error 0 at a = 8.067 A (published 8.083, -0.2 %), detected in 227
  (`tests/integration/test_composition_free_known_answer.py`, slow, 72 s).
- **End members under the Al-O-N constraints** (Al CN [4, 6], O [2, 4], N [3, 4], density
  [3.26, 4.10] g/cm^3), `mizuho check`: corundum 5.2e-4 (its long Al-O 1.972 A is 0.017 A past
  the derived band end 1.955 A of a [4, 6] range, whose s0 is taken at CN 5); wurtzite AlN
  4.5e-6 (a 3.112, c 4.982 A, u 0.382, not read from a source in this environment). Both are
  below the zero-error threshold 1e-3.
- `export_classes.py` on the TiO2 records reproduces `runs/classes/tio2` byte for byte.
- N was missing from `data/atomic_weights.py`; added at 14.007 (IUPAC 2021 abridged).
