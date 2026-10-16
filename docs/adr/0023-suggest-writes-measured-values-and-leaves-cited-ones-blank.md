# `suggest` writes measured values and leaves cited ones blank

ADR-0013 reserved `mizuho suggest ref.cif` until ADR-0007's band rule existed; uniqueness plan
phase E implemented it. What `suggest` may write is bounded by ADR-0004, which forbids re-deriving
a coordination number from our own model ("the model would choose its own exam questions"), and by
ADR-0017, which takes a volume from a measured density and never from a cell.

## Decision

- **Scope: fixture authoring.** `suggest` mechanises the rules that have so far been applied by
  hand, a step that has gone wrong before (spinel's origin choice, topaz's auto-injected cutoff):
  - cutoffs at the first/second shell mid-gap (ADR-0004);
  - bond bands: the reference span widened by 2%, rounded outward (ADR-0007);
  - angle bands as measured after `ignore_trans_above`, rounded outward to 5 deg (ADR-0004's
    amendment);
  - a warning when any atom lies within cutoff +/- 3w (ADR-0015);
  - a bond-valence tolerance only where the reference exceeds 0.1 v.u., with the reason beside it
    (ADR-0016).
- **`cn_min` / `cn_max` are written as `null`**, with the measured integer in a comment beside them
  (`cn_min: null  # measured 6 at ref; cite before use`). Config validation rejects a null
  coordination bound (added with `suggest`; it does not today), so the fixture does not run until a
  citation fills it.
- **Volume bounds only from `_exptl_crystal_density_meas`** when the CIF carries it. Otherwise the
  bounds are left blank, with a note that D(calc.) is not used.
- **Not for prediction by analogy** (one compound's constraints applied to another composition).
  That is a different feature, discussed when it is needed.

## Consequences

- The inconvenience is deliberate: a measured coordination number is available and is not
  written as a value.
- Implemented after ADR-0020's measurements.
