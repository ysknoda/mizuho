# The CLI is a set of subcommands

`mizuho input.yaml` did one thing — predict a structure — and chose between single-space-group,
automatic-space-group and fixed-assignment modes through the YAML. Verification needed four more
things, and each was written as a throwaway script more than once during this project: evaluating
the objective at a published structure, optimising from a published topology and comparing the
result, listing Phase 1's ranking, and looking up Wyckoff positions in mizuho's own setting (the
origin-choice trap behind the spinel and anatase cases). They belong in the tool.

## Decision

| subcommand | purpose | replaces |
| --- | --- | --- |
| `run cfg.yaml` | predict structures; everything the old command did, plus `--format poscar\|cif` and a report of the detected space group, lattice-bound contacts and integer-versus-soft coordination disagreements | `mizuho input.yaml` |
| `check cfg.yaml ref.cif` | evaluate the objective at a reference structure: per-term breakdown, per-site soft and integer coordination numbers, detected space group. No optimisation | `diag.py`, `verify_cifs.py` |
| `fit cfg.yaml ref.cif` | fix the reference's topology, optimise lattice and coordinates, compare the result with the reference: lattice deviation, bound contacts, and a warning when the point reached scores below the reference (an impostor, ADR-0007) | `realtopo.py`, `uniq.py` |
| `candidates cfg.yaml` | run Phase 1 only and list the ranking after merging equivalent assignments, with free-parameter counts, strata and the rank spread over seeds | `rankpos.py`, `strata.py` |
| `wyckoff SG` | list a space group's Wyckoff positions in mizuho's setting: letter, multiplicity, free parameters, representative coordinates, origin choice, and which letters are equivalent under the normalizer | manual lookups |
| `suggest ref.cif` | generate constraints from a reference structure | name reserved only |

- `mizuho input.yaml` stops working. There is no fallback that treats a first argument ending in
  `.yaml` as an implicit `run`: that is exactly the compatibility special case ADR-0002 declines to
  carry, and the package is an alpha release.
- `suggest` waits for `docs/plan-objective-uniqueness.md`, because the band-width rule it would
  encode (ADR-0007) is not implemented yet. Reserving the name now keeps the command set stable.

## Consequences

- `check` and `fit` share the CIF reader and the setting mapping of ADR-0010, so most of their cost
  is paid once.
- `fit` is the user-facing form of ADR-0005's global-minimum gate, and the entry point for fitting
  structure parameters in a space group the test fixtures do not cover.
