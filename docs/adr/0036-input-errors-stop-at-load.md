# Input errors stop at load time, one `ERROR:` line each

Auditing which inputs have no default (2026-10-27) found that a wrong config could look like a
finished run. A `bond_constraints` or `repulsion_constraints` entry without `distance_min` (or
another required key) raised a KeyError inside every candidate's optimisation; the driver's
`except Exception` turned each into "optimisation failed", printed only under `--verbose`, and
`run` exited 0 with error 1e10 for every candidate. A `volume_constraint` with only `volume_min`
silently used the automatic band instead. Missing keys in coordination, angle and hybridisation
entries, and `coordination_numbers` without an anion charge, stopped with raw tracebacks.

## Decision

- **Required entry keys are checked at load**, in `objective.penalties.validate_constraint_entries`,
  called by `cli._load_config` beside `reject_removed_keys` for every subcommand. The keys without
  a default (`REQUIRED_ENTRY_KEYS`): bond `atom1 atom2 distance_min distance_max`; coordination
  `center_atom neighbor_atom cn_min cn_max cutoff_distance`; angle `atom_center atom_neighbor
  angle_min angle_max`; repulsion `atom1 atom2 distance_min`; hybridisation `element
  hybridisation`. Each list must be a list of mappings. `volume_constraint` states both bounds or
  neither. Every problem is listed in one message, e.g.
  `ERROR: bond_constraints[0] is missing distance_min`.
- **Opt-in switches are booleans.** `auto_repulsion` (ADR-0040) is read as `is True`, so any
  other value (`"yes"`, `1`) would leave it off without a word; at load it must be true or false
  (`BOOLEAN_SWITCHES`).
- **`errors.ConfigError(ValueError)` marks an input error.** The derived-constraint checks
  (ADR-0028), the composition-free checks (ADR-0033), the density and void bands, the removed-key
  check, `topology_grouping` and the backend checks raise it; `cli.main` prints it as one `ERROR:`
  line and exits 1. Any other exception is a defect and keeps its traceback. Subclassing
  ValueError keeps every caller and test that catches ValueError working.
- **`run` derives the constraints once before searching**, so a derivation error stops at once,
  not at the first group with an assignment (hours into an exhaustive run). Without a
  composition, one formula unit of each element stands in.
- **The optimiser no longer absorbs input errors and defects.** `optimize_assignment` re-raises
  ConfigError, KeyError, TypeError, AttributeError, NameError and ImportError: they fail every
  candidate alike and are not a candidate that did not converge. Any other exception still marks
  that one candidate as failed, now always reported with its type, not only under `--verbose`.
  A long run that hits one stops; its record (ADR-0026) resumes it after the fix.

## Verified

`tests/unit/test_config_validation.py`: each required key of each list, both volume bounds, wrong
shapes, every fixture valid, `run` stopping before Phase 1 on a missing key and on a derivation
error, and the optimiser re-raising KeyError, TypeError and ConfigError while still reporting a
numerical failure as one failed candidate. Probed by hand through `python -m mizuho run`: the
eleven cases of the audit (the six above plus unknown backend, composition-free search without
`max_atoms`, inverted density band, coordination entry without its cutoff, hybridisation entry
without its state) all end with `ERROR:` and exit 1.
