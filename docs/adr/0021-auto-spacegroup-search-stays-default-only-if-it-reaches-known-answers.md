# The automatic space-group search stays the default only if it reaches its known answers

Measured at 0.1.0a3 (the plan's Status, 2026-10-01), judged on detected symmetry at 1e-3 A:
NaCl -> 99, CsCl -> 51, Si -> 123, ZnO -> 123, TiO2 -> 47. None is the known answer (ADR-0010).
`tests/integration/test_auto_spacegroup_known_answers.py` marks only `cscl` and `si` xfail, so
`nacl`, `zno` and `tio2` would fail outright under `--run-gates`, against ADR-0005's rule that a
failing gate is recorded as xfail. Neither the test nor its fixtures has run since, while
ADR-0014 to ADR-0019 changed the objective under them: the gray band and compact support (which
close the mechanism `si`'s xfail reason describes), bond valence, volume and topology classes.

Mode B is what a config without `spacegroup` gets. A miss there is a plausible structure in the
wrong group. The detected group is printed, but nothing tells the user it is wrong.

## Decision

- Re-run the five known-answer tests on the current code first, and bring the xfail table and the
  plan's Status in line with the result.
- **Rule, fixed before that run: if any of the five misses its known answer, Mode B becomes
  explicit.**
  - A config with neither `spacegroup` nor `mode: auto_spacegroup` is rejected with an error naming
    this ADR.
  - `mode: auto_spacegroup` still runs, marked experimental, and `run` prints that its known-answer
    tests do not pass.
  - No compatibility flag (ADR-0002, ADR-0013); this is an alpha release.
- If all five pass, Mode B stays the default.

The strict rule was chosen over keeping the default when only NaCl and CsCl pass: a polytype family
counts as a pass already (ADR-0010), so a miss on Si, ZnO or TiO2 is not a polytype ambiguity.

## Consequences

- L1 of step 7 (ADR-0020) starts only after this rule has been applied.
- If Mode B becomes explicit, the README's description of it moves under an experimental heading.
