# mizuho

> **Alpha release.** This project is under active development.
> Interfaces, behavior, and output formats may change without notice
> between releases, and it has not yet been broadly tested. Do not
> rely on it for production use.

Automatic crystal structure prediction from composition, chemical-bonding
constraints (bond distance / bond angle), and space-group / Wyckoff-position
constraints.

Given a target composition (e.g. `NaCl`), a set of bond/angle/coordination
constraints, and either a specific space group or a request to search over
all 230 space groups, mizuho searches over valid Wyckoff-site assignments
and numerically optimizes the lattice constants and free atomic coordinates
to satisfy the constraints, producing candidate structures in VASP POSCAR
format.

## Installation

```bash
pip install .
```

That is the complete runtime install. Every extra below is optional: each
enables one additional opt-in backend or accelerator, and without it mizuho
falls back to its default pure-NumPy/CMA-ES path rather than failing.

```bash
# Optuna TPESampler, for optimization.backend = "tpe"
pip install ".[bayesian]"

# Optuna GPSampler (Gaussian-process Bayesian optimisation), for
# optimization.backend = "gp" / "bayesian". Pulls in torch.
pip install ".[gp]"

# emcee, for optimization.backend = "mcmc"
pip install ".[mcmc]"

# numba JIT for the pairwise-distance kernel -- pure speed, no new options
pip install ".[numba]"

# JAX analytic gradient for optimization.refine_jac = true
# (measured slower on every current fixture -- see the section below)
pip install ".[jax]"

# The test suite, including spglib-based validation of the built-in
# Wyckoff position database
pip install ".[test]"
```

### About the Wyckoff position database

`mizuho.data.WYCKOFF_DB` (1,731 entries across all 230 space groups) is not
hand-transcribed and depends on no external Wyckoff-position table. Every
position is derived first-principles by `scripts/generate_wyckoff_db.py`
from mizuho's own true symmetry-operation table
(`mizuho.data.symmetry_ops`, generated once from spglib by
`scripts/generate_symmetry_ops_table.py`):

1. For each symmetry operation, solve for its fixed-point subspace exactly
   over the integers (Smith Normal Form), which correctly enumerates cases
   like a 2-fold rotation about z through the origin also fixing the
   parallel axis at (1/2, 0, z) via the lattice's periodicity.
2. Close this set under pairwise intersection to find every possible
   site-symmetry-fixed subspace (e.g. a point fixed by two independent
   mirrors but by neither mirror's rotation part alone).
3. Deduplicate subspaces related by a group operation -- checked by
   transforming one subspace's point AND free directions and testing for
   set-equality with the other, which (unlike checking a single matched
   point) correctly catches equivalences that also reparametrise the free
   coordinate, e.g. a centring translation mapping line (x,0,0) onto line
   (x,1/2,0).
4. Confirm each surviving subspace's true multiplicity and canonical
   Wyckoff letter through an independent spglib re-analysis (build a trial
   structure from its orbit, plus a generic "anchor" orbit so the trial
   structure has no accidental extra symmetry, and check spglib recovers
   the same space group and a single consistent letter).

Every one of the 1,731 entries is cross-checked against this same spglib
re-analysis in `tests/validation/test_wyckoff_db_vs_spglib.py`. `spglib`
itself stays a test/dev-only dependency -- installed `mizuho` never needs
it (or any other symmetry library) at runtime.

## Quick start

Create a YAML input describing the target composition, space group, and
constraints:

```yaml
# nacl.yaml -- rock salt structure, space group 225 (Fm-3m)
composition: NaCl
spacegroup: 225
spacegroup_symbol: Fm-3m

ionic_charges:
  Na: 1
  Cl: -1

charge_tolerance_min: 0
charge_tolerance_max: 0

lattice_bounds:
  a: [4.5, 6.5]

bond_constraints:
  - atom1: Na
    atom2: Cl
    distance_min: 2.5
    distance_max: 3.2
    weight: 1.0

tree_search:
  max_sites: 4
  max_formula_units: 4

optimization:
  n_restarts: 8
  cma_maxiter: 400
```

Then run:

```bash
mizuho nacl.yaml --output nacl.vasp --seed 42
```

This writes the best candidate structure to `nacl.vasp` (VASP POSCAR
format) and prints a ranked summary of all candidates considered, along
with a per-constraint error breakdown for the winner.

## How it works

Two execution modes, selected by whether `spacegroup` is given (`mode` may
also be set explicitly to `single_spacegroup` / `auto_spacegroup` for
clarity; it must then agree with whether `spacegroup` is present):

1. **Mode A -- single space group** (`spacegroup: <1-230>`): run the
   Wyckoff-assignment search and numerical optimisation directly against
   the given space group.
2. **Mode B -- automatic space-group search** (`spacegroup` omitted): a
   hierarchical search over crystal system -> point group ->
   symmorphic/non-symmorphic -> individual space group selects promising
   candidates before handing each one to mode A's pipeline. Selection uses
   PUCT (prior-weighted UCB, as in AlphaZero-style search): each node's
   prior blends a chemical-category x crystal-system heuristic with
   symmorphic / centrosymmetric / known-common-space-group weights
   (`mizuho.data.point_group_priors`), and a leaf space group is confirmed
   with a real (but budget-limited, successive-halving) CMA-ES run only
   after enough PUCT visits -- so budget concentrates on the
   highest-promise part of the tree without ever fully excluding the rest.
   Set `spacegroup_search.strategy: flat` to use the original 3-stage
   screening instead.

Once a space group is fixed (by either mode), the shared pipeline is:

3. **Wyckoff-assignment search**: mizuho enumerates valid assignments of
   elements to Wyckoff positions that satisfy the target composition ratio
   and (for ionic compounds) charge neutrality, using a best-first
   branch-and-bound search ordered by a cheap surrogate score (charge
   balance + hard-sphere packing fraction). Set
   `tree_search.strategy: exhaustive` to fall back to a plain exhaustive
   DFS (kept as a correctness oracle; both explore the same search space).
4. **Numerical optimization**: for each candidate assignment, the free
   lattice constants and free Wyckoff-position parameters are optimized
   (CMA-ES global search, via the `cma` package, + L-BFGS-B local
   refinement) against an objective function built from the bond-distance /
   bond-angle / coordination-number / repulsion / charge-balance /
   cell-volume constraints supplied in the YAML input. Independent
   candidates can be optimised in parallel via
   `mizuho.optimize.optimize_assignments_parallel`.
5. Candidates are ranked by total constraint-violation error, and the
   top-N structures are written out as VASP POSCAR files.

## Configuration reference

See the YAML keys used in the quick-start example above. The most
commonly used top-level keys are:

| Key | Purpose |
|---|---|
| `composition` | Target chemical formula, e.g. `"BaTiO3"` |
| `spacegroup` | Space-group number (1-230); omit for auto-search (mode B) |
| `mode` | Optional explicit `single_spacegroup` / `auto_spacegroup`; must agree with whether `spacegroup` is set |
| `ionic_charges` | `{element: charge}` for ionic-mode searches |
| `lattice_bounds` | Per-axis `[min, max]` bounds for lattice constants |
| `bond_constraints` | List of `{atom1, atom2, distance_min, distance_max, weight}` |
| `angle_constraints` | List of `{atom_center, atom_neighbor, angle_min, angle_max, weight, exclude_neighbor_pairs}` |
| `coordination_constraints` | List of `{center_atom, neighbor_atom, cn_min, cn_max, cutoff_distance}` |
| `tree_search` | Wyckoff-search budget: `max_sites`, `max_formula_units`, `max_candidates`; `strategy: exhaustive` for the DFS oracle |
| `spacegroup_search` | Mode-B tuning: `top_sg_count`, `mcts_budget`, `confirm_after_visits`, `crystal_system_hint`; `strategy: flat` for the original 3-stage screening |
| `optimization` | CMA-ES / L-BFGS-B settings: `n_restarts`, `cma_maxiter`, `seed`, ... |
| `auto_composition` | Set `true` for fully automatic composition + structure discovery |
| `composition_pool` | List of composition strings to search over |

### Neighbour/partner element specification (`atom2`, `neighbor_atom`, `atom_neighbor`)

These fields accept three forms:
- A single element symbol, e.g. `O` -- matches only that species.
- An explicit list, e.g. `[Zn, As]` -- matches exactly those species (filtered to ones actually present); an escape hatch for when charge-sign inference below isn't precise enough.
- The wildcard `X` -- resolves automatically to every present species whose ionic charge has the **opposite sign** from the constraint's own center/`atom1` (e.g. an O center with charge -2 resolves `X` to every cation present, never to O itself or any other anion). If `ionic_charges` is absent, or the center's own charge is 0/undefined, `X` falls back to "every other species present" (self excluded, sign ignored). `coordination_constraints` already pool every matched neighbour species into a single combined coordination-number count, so `neighbor_atom: X` on an anion center gives "total bonds to any cation" for free, with no extra configuration.

`angle_constraints` additionally accept `exclude_neighbor_pairs`, a list of 2-element species pairs to drop from angle-triple evaluation regardless of how the neighbour set was resolved, e.g.:

```yaml
angle_constraints:
  - atom_center: O
    atom_neighbor: X
    angle_min: 80.0
    angle_max: 160.0
    exclude_neighbor_pairs: [[H, H]]   # skip H-center-H angles (they'd model a water molecule)
```

### Automatic anion coupling

Whenever `ionic_charges` is set and some cation-centred `coordination_constraint`/`angle_constraint` references an anion as its neighbour (directly, via `X`, or via an explicit list) but that anion has no constraint of its own as a center, mizuho auto-injects a default anion-centred pair (`neighbor_atom`/`atom_neighbor: X`) so the cation polyhedra sharing that anion are coupled to a mutually consistent geometry instead of drifting apart independently. Tuning/opt-out keys:

| Key | Default | Purpose |
|---|---|---|
| `auto_anion_coupling` | `true` | Disable the whole feature |
| `auto_anion_coupling_angle` | `true` | Keep the coordination constraint but skip the angle one |
| `auto_coupling_cn_range` | `[2, 4]` | Override the injected `cn_min`/`cn_max` |
| `auto_coupling_angle_range` | `[80, 160]` | Override the injected `angle_min`/`angle_max` |
| `auto_coupling_weight` | `1.0` | Override the injected weight |
| `auto_coupling_exclude_hh` | `true` | Auto-add `exclude_neighbor_pairs: [[H, H]]` to the injected angle constraint whenever H is one of the coupled cations (an H-center-H angle would model a water molecule, not the intended cation-anion-cation bridging geometry); set `false` to disable |

### Penalty normalisation

Every penalty term is a squared deviation in its own raw physical unit (bond: Angstrom, angle: degrees, coordination number: a dimensionless count, ...), summed together with no scale reconciliation between them. In practice this lets the coordination-number term numerically dominate the total error (measured at 65-99% of the total across this project's own fixtures), regardless of the nominal per-constraint `weight`. Setting the top-level key `normalize_penalties: true` divides each term's deviation by a natural physical scale for that term (the target band's own midpoint for bond/angle/coordination-number/volume; the cell's total absolute ionic charge for the charge-balance term) before squaring, so all terms become comparable relative deviations. Default `false` reproduces every prior release's exact numeric behaviour; `hybridisation_constraints` and the overlap term are unaffected either way (overlap is already relative by construction; hybridisation is excluded because no fixture has ever exercised it with a nonzero value, see the mizuho-py-wychoff-position plan's phase I-1).

### JAX-accelerated Stage 2 gradient (measured: not recommended)

Setting `optimization.refine_jac: true` supplies Stage 2's local refinement with an exact analytic gradient for the coordination-number and volume penalty terms (via [JAX](https://github.com/google/jax) autodiff), combined with scipy's usual finite-difference gradient for every other term. Requires the optional `jax` extra (`pip install mizuho[jax]`); raises a clear `ImportError` otherwise. Not supported together with `hybridisation_constraints` (raises `ValueError`) since that term's hard neighbour cutoff and discontinuous `ang > 150 deg` branch have not been validated for JAX differentiation.

**Measured, not recommended for general use.** The gradient itself is exact (bit-identical final error vs. the finite-difference default on every fixture tested). But `scripts/benchmark_jax_gradient.py`'s real end-to-end Stage-2 comparison found it **2-8x slower** on all three fixtures tested (adamite, spinel_mgal2o4, topaz), because Stage 1's CMA-ES already converges close enough that Stage 2 typically needs only 0-1 L-BFGS-B iterations -- leaving no room to amortise the gradient function's own one-time JIT-compilation cost (~0.7-2s, measured) against per-iteration savings. It is kept available as an opt-in in case a future problem needs many more Stage-2 iterations than this project's own fixtures do (where the calculus could differ), but default `false` changes nothing and there is currently no fixture on which enabling it helps -- see the mizuho-py-wychoff-position plan's phase I-2 for the full measured record.

## Testing

```bash
pip install -e ".[test]"
pytest
```

The end-to-end tests that run a real search/optimisation pipeline are marked
`slow`; `pytest -m "not slow"` skips them.

## License

MIT -- see [LICENSE](LICENSE).
