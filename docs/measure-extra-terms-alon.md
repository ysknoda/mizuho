# Would extra objective terms prune the Al-O-N zero-error classes? (measurement, 2026-11-01)

The Al-O-N composition-free run (ADR-0033) left 2,565 zero-error topology classes. Its
constraints are loose: Al CN [4, 6] has no angle band, anion bond-valence sums are reported
only, the anion-centred band stops at 150 deg, and no anion-anion repulsion exists (the
hard-sphere term acts only below r_O + r_O = 1.32 A). This note measures, without rerunning the
search, whether four candidate terms would prune the classes towards the physically plausible
ones, judged by the MACE pre-screen (ADR-0035).

**Answer: none of them does.** After local re-optimisation every term (and their sum) removes
low-energy classes at about the rate it removes high-energy ones. The share of kept classes
within 100 meV/atom of the tie line stays at the unfiltered 2.0 % (1.7-2.3 %). The sum of all
four terms raises it to 4.4 %, but only by removing 47 of the 52 low-energy classes and 18 of
the 19 zero-error stage-1 picks. No ADR is drafted; mizuho's defaults are unchanged.

## Method

- **Entries.** Every zero-error entry of runs/alon/alon.jsonl (4,735; error <= 1e-3) is rebuilt
  with `mizuho.cli._entries_from_record` and its composition's config. The recomputed objective
  equals the recorded error exactly. Classes are numbered as `scripts/export_classes.py`
  numbers them; all 2,568 rows of classes.csv match in entry count, error and space group.
- **Terms**, each in the form, weight (1.0) and normalisation of the existing term it extends
  (`scripts/measure_extra_terms.py`):
  1. `bvs_f`: anion bond-valence sums, |BVS - V| beyond f |V| (O 2, N 3), scaled by |V| and
     averaged per element. This is the cation term of ADR-0016/0028 applied to anions;
     f = 0.10-0.30.
  2. `aa_d`: anion-anion repulsion O-O, O-N, N-N below d = 2.0-2.5 A (`repulsion_constraints`).
  3. `xalx`: X-Al-X bands by each Al's actual CN, using ADR-0028's bands: CN 4 [95, 125],
     CN 6 [70, 110], CN 5 none. Angles above 150 deg are dropped, as compute_error does;
     `xalx_t4` also holds a tetrahedron's angles above 150 deg.
  4. `axa_a`: the automatic anion-centred band [80, a] over every angle (ignore_trans_above
     180 instead of the default 150); a = 150, 160 (the configured band), 170.

  `aa`, `axa` and single-CN `xalx` agree with compute_error's own increase to <= 1e-9 on
  random entries and in the tests. `bvs` reproduces the cation penalty exactly.
- **Survival.** An entry survives when recorded error + penalty <= 1e-3. A class survives
  when any of its zero-error entries does.
- **Local re-optimisation (`reopt`).** A class that every recorded point fails is
  re-optimised from its best point: L-BFGS-B on objective + term, bounds and maxiter 200 as in
  Phase 2. It survives if it reaches <= 1e-3 in the same topology class. This was run for
  every removed class of seven term sets (15,007 runs).
  - The re-optimisation is local, has one start and stays in the recorded space group. Its
    survival is therefore a lower bound and its removal an upper bound on what a rerun would
    remove.
- **MACE labels** (runs/alon/mlip, complete ledger of 2,570):
  - excluded for anion-anion < 2.0 A (419 classes) or non-convergence (19);
  - saddle (1,857);
  - energy above the tie line > 150 meV/atom (1,855, not excluded) or <= 100 meV/atom
    (52, not excluded);
  - stage-1 picks: 19 zero-error classes; the 20th pick is an Al7O9N near-miss.

## End members

Corundum (corundum_H.cif) scores 5.2e-4 and wurtzite AlN 4.5e-6, as in ADR-0033. Both pay
exactly 0 under all 17 term variants.

| | corundum | wurtzite AlN (a 3.112, c 4.982 A, u 0.382; UNCERTAIN, not read from a source here) |
|---|---|---|
| anion CN, BVS | O 4, 1.99 | N 4, 3.02 |
| shortest anion-anion | O-O 2.526 A (margin 0.026 A at aa_2.5) | N-N 3.071 A |
| X-Al-X | CN 6, cis 79.7-101.2 deg | CN 4, 108.1-110.8 deg |
| largest Al-X-Al | 132.2 deg | 110.8 deg |

## Result

Classes kept out of 2,565, and classes removed in each MACE category.

| term(s) | kept at recorded points | kept after re-optimisation | removed: anion-anion excl. /419 | saddle /1857 | > 150 meV /1855 | <= 100 meV /52 | picks /19 | % kept <= 100 meV |
|---|---|---|---|---|---|---|---|---|
| none | 2565 | 2565 | 0 | 0 | 0 | 0 | 0 | 2.0 |
| bvs_0.10 | 145 | 1522 | 202 | 779 | 734 | 20 | 6 | 2.1 |
| bvs_0.15 | 324 | 2368 | 41 | 155 | 132 | 4 | 2 | 2.0 |
| bvs_0.20 | 860 | 2459 | 22 | 88 | 64 | 3 | 2 | 2.0 |
| aa_2.3 | 223 | 1108 | 335 | 1129 | 964 | 33 | 12 | 1.7 |
| xalx | 482 | 666 | 309 | 1358 | 1379 | 37 | 18 | 2.3 |
| axa_160 | 910 | 1581 | 157 | 739 | 719 | 20 | 9 | 2.0 |
| all four (bvs_0.20 + aa_2.3 + xalx + axa_160) | 4 | 114 | 415 | 1798 | 1766 | 47 | 18 | 4.4 |

Removal counts are after re-optimisation. Recorded-point numbers for every threshold and
every pair and triple are in the [generated tables](measure-extra-terms-alon-tables.md).

### At the recorded points

The recorded points remove most classes under every term, but say little.

- 1,949 classes have an anion pair closer than 2.0 A at every recorded zero-error point, and
  532 classes closer than 1.4 A.
- Every term removes most stage-1 picks at these points (e.g. aa_2.3 18/19, bvs_0.20 13/19).
- The kept classes' median MACE energy stays at that of all classes (257-285 against 278
  meV/atom).
- The zero-error region of a class is wide, and the optimiser stops anywhere in it. Most
  classes have another point that meets the term: bvs_0.20 1,599 of 1,705, aa_2.3 885 of 2,342.

### Per term

- **bvs** (term 1) hardly prunes at 15-20 %. At 10 % it removes 40 % of the classes, but
  evenly: 20/52 low-energy and 734/1855 high-energy.
  - On the MACE-relaxed structures, 10 % is too strict for real low-energy ones: only 0.60 of
    those <= 75 meV/atom pass it. 15 % passes all of them (n = 20).
- **aa** (term 2) removes 80 % of the classes MACE later excluded for anion-anion bonds (335
  of 419), but also 63 % of the low-energy classes (33/52) and 12 of the 19 picks.
  - The relaxed low-energy structures all keep anion-anion >= 2.41 A, so 2.3 A is not too
    strict for them.
  - In 1,191 of the 1,457 removed classes the topology is kept but the penalty cannot be
    removed inside the recorded space group. Those classes relax away from their short
    contacts under MACE.
- **xalx** (term 3) is the strongest single term (666 kept), and the least selective: it
  removes 18 of the 19 picks.
  - ADR-0028's tetrahedral band [95, 125] rejects edge-sharing AlX4 tetrahedra; relaxed
    low-energy Al3O3N structures have X-Al-X down to 87 deg.
  - It passes only 0.85 of the relaxed structures <= 75 meV/atom.
- **axa** (term 4, angles above 150 deg evaluated) removes 984 classes, again evenly (20/52
  low-energy, 719/1855 high-energy, 9/19 picks).
  - 875 of the 984 have a symmetry-fixed linear bridge: an anion angle of exactly 180 deg at
    every recorded point. Re-optimisation cannot bend it inside the recorded group.
  - MACE's symmetry-free relaxation bends these bridges. All 20 removed low-energy classes
    have 180 deg at their recorded points, but 0-0.009 penalty on their relaxed structures.

### Conflict between term 1 and O CN [2, 4]

- **The bound.** Inside the derived Al-O band (s0 = 3/5 at the CN [4, 6] midpoint,
  +/- 0.16 v.u.) one bond carries at most 0.76 v.u. An O at CN 2 therefore sums to at most
  1.52 v.u., 24 % below 2. N at CN 3 is bounded the same way: 2.28 against 3.
  (`test_two_coordinated_oxygen_cannot_reach_its_valence`.)
- **At the recorded points.** O at CN 2 has median BVS 1.34 (max 1.80, reached by bonds
  leaking below the band within the 1e-3 threshold); N at CN 3 has median 2.04.
- **A hard criterion** at any tolerance below 24 % would forbid O CN 2 and N CN 3 outright.
  - That would cut the user's O [2, 4] to [3, 4], and N [3, 4] to 4.
  - 1,152 classes have O at CN 2, and 473 have N at CN 3, at every recorded point.
- **The soft penalty** is a per-element mean with a 1e-3 threshold, so it dilutes the
  violation instead.
  - One CN-2 O at 1.52 v.u. costs (0.48 - 2f)^2 / 4 / N(O): 0.0196 / N(O) at f = 0.10,
    0.0081 / N(O) at 0.15, 0.0016 / N(O) at 0.20.
  - At the re-optimised points every CN-2 O still lies beyond the tolerance (BVS 1.66-1.80).
    Yet 1,052 of the 2,044 classes recovered at 15 %, and 455 of the 1,377 recovered at
    10 %, keep CN-2 O.
- **The data do not single out O CN 2.**
  - Classes with O at CN 2 at every point have median MACE energy 281 meV/atom, against 276
    for the others; 28 of them are within 100 meV/atom (others 24).
  - On relaxation those O usually become 3-coordinated. No relaxed structure within 75
    meV/atom has an O at CN 2, and only 7 of 174 O sites in 75-100 meV/atom do.
  - N at CN 3 is somewhat enriched at high energy: 8 of its 473 classes are low-energy,
    against 44 of the other 2,092.

## Recommendation

- **Do not add any of the four as an objective term for this search**, at any threshold
  measured.
  - Each one removes the low-energy classes in about the same proportion as the rest.
  - Most picks are lost under aa, xalx and their sum.
  - The common cause is that the plausible classes reach their low energies only after MACE
    breaks the recorded symmetry. Their recorded high-symmetry geometries (short anion
    contacts, linear bridges, distorted polyhedra) are exactly what the terms reject.
- **Keep the pruning after relaxation, as ADR-0035 does.** MACE's anion-anion exclusion and
  the energy ranking already remove the bad classes that the terms would remove, and they do
  so without losing the good ones.
- **If the 2,565-class count must come down before relaxation**, measure a symmetry-free
  local relaxation of each recorded point (e.g. in P1) under the terms, rather than adding the
  terms at the recorded symmetry. This was not measured here.
- **Not measured: a rerun of the search with a term.** It could find classes in groups the
  record does not hold, so the removal numbers above are upper bounds. That cannot make a
  non-selective term selective.

Reproduce:

```
python scripts/measure_extra_terms.py score  --config runs/alon/alon.yaml --record runs/alon/alon.jsonl --out OUT -j 6
python scripts/measure_extra_terms.py verify --config runs/alon/alon.yaml --record runs/alon/alon.jsonl --n 200
python scripts/measure_extra_terms.py refs   --config runs/alon/alon.yaml
python scripts/measure_extra_terms.py reopt  --config runs/alon/alon.yaml --record runs/alon/alon.jsonl --scores OUT/scores.jsonl --terms xalx --out OUT/xalx.jsonl -j 6
python scripts/measure_extra_terms.py report --config runs/alon/alon.yaml --record runs/alon/alon.jsonl --scores OUT/scores.jsonl --classes runs/alon/classes --mlip runs/alon/mlip --reopt OUT/*.jsonl --out OUT
```

Cost: scoring 64 s on one core; re-optimisation of the seven sets about 38 core-hours (one start
per class, 1-30 s each).
