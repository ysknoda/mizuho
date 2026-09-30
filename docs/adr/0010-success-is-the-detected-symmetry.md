# A search succeeds when the final structure's *detected* symmetry is right

When mizuho searches over space groups, "did it find the right one?" has no simple answer, because
the objective sees only bonding geometry. Three things tie with the correct answer by construction:

- **Subgroups.** A structure written in a subgroup of its true space group has every degree of
  freedom it had before and more, so it scores at least as well. NaCl described in P1 can reach
  error zero.
- **Enantiomorphs.** P3₁21 and P3₂21 describe mirror images. Bond lengths, angles and coordination
  numbers are identical, so the two always score the same. Only anomalous scattering tells them
  apart, and mizuho has no diffraction data.
- **Polytypes.** First-shell constraints cannot separate structures that differ only in how
  identical coordination units are stacked or linked: cubic and hexagonal diamond for Si
  (227 / 194), wurtzite and zinc blende for ZnO (186 / 216), rutile and anatase for TiO₂
  (136 / 141).

Judging success by the space-group label the search returned would call a correct structure found
in a subgroup a failure, and has no answer at all for the other two.

## Decision

**Success is judged on the optimised structure, not on the label.** After optimisation, the
structure's actual space group is detected. The search succeeds if that detected group is the true
one; either member of an enantiomorph pair counts; and for a composition whose constraints admit a
known polytype family, any member of that family counts.

**A higher detected symmetry is acted on, not only reported.** If the structure optimised in G is
detected in a supergroup G′, it is symmetrised to G′, mapped onto a G′ Wyckoff assignment in
mizuho's own setting, and re-optimised there. The answer returned is then a genuine G′ optimum,
with an error that belongs to the symmetric structure.

**spglib becomes a required dependency.** Every optional extra so far (numba, jax, optuna, emcee)
is an accelerator or an extra backend: leaving it out changes speed, never the result. Symmetry
detection is now part of the verdict, so an environment without it would reach a different
conclusion from the same input. Detection needs a search over origin shifts and cell
transformations (P1 to the conventional Fm-3m cell, for instance), which is exactly what spglib
does; reimplementing it would be a large project with no benefit. It was already in the `test`
extra.

**Tolerance is reported as a ladder and the threshold is measured, not chosen.** The detected group
is always reported at 1e-4, 1e-3, 1e-2 and 1e-1 A. The tolerance at which success is judged is set
from the coordinate noise the optimiser actually leaves on reference-structure runs, with a margin,
in the same spirit as ADR-0004 measuring cutoffs rather than choosing them. It is not fixed here
because it has not been measured.

## Consequences

- The mapping "structure in any setting → mizuho's Wyckoff assignment" is needed twice: here, for
  the re-optimisation in G′, and for reading a CIF written in another setting or origin choice.
  Both use spglib's dataset for the Hall number mizuho's tables were built from (the first Hall
  number of each space group). One implementation serves both.
- Accepting a polytype family is a stated limit of first-shell information, not a relaxation of
  ADR-0007. ADR-0007 requires the published structure to be the only minimum and says a tie with
  another legitimate structure means the input lacks information. Here it genuinely does:
  distinguishing polytypes needs second-shell information, which is out of scope for now. As with
  Z in ADR-0008, the ambiguity is reported rather than hidden.
- Enantiomorph handling is tested exactly by inverting a structure (x, y, z → −x, −y, −z maps
  P3₁21 onto P3₂21) rather than by comparing two independent refinements, which would confound
  handedness with refinement noise.
