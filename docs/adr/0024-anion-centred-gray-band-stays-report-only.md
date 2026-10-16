# The anion-centred gray band stays report-only

ADR-0014 left open whether anion-centred shells join the gray-band penalty, pending non-oxide
references. Those now exist (fluorite, pyrite, sphalerite), and ADR-0015 measured the condition:
of the thirteen references only diopside's O(X) has an atom inside its band (a 0.446 A gap against
a 0.6 A band), which the penalty would charge at the published structure (4.6e-4).

The reason to add it was two impostors, adamite's and diopside's, that the anion version separates
and the cation version does not. ADR-0019 removed that reason: diopside's points relax to the
published structure under DFT. Adamite's were recorded against the 1937 reference, which ADR-0025
replaces, and the records are retired.

## Decision

- Not added. `gray_band_report_all_centres` stays in `error_breakdown` and in the `check`, `fit`
  and `run` reports.
- Reopened only by a point DFT confirms to be a different structure that the anion version alone
  separates.
