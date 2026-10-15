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

# The test suite
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
re-analysis in `tests/validation/test_wyckoff_db_vs_spglib.py`. The database
itself is shipped as generated tables; `spglib` is nevertheless a required
runtime dependency, because detecting the symmetry of an optimised structure
is part of how a search is judged
([ADR-0010](docs/adr/0010-success-is-the-detected-symmetry.md)).

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
mizuho run nacl.yaml --output nacl.vasp --seed 42
```

This writes the best candidate structure to `nacl.vasp` (VASP POSCAR
format; `--format cif` writes a CIF in the structure's space group) and
prints a ranked summary of all candidates considered, a per-constraint
error breakdown for the winner, its detected space group at 1e-4 to 1e-1 A,
and warnings for any lattice parameter sitting on a `lattice_bounds` edge
or any coordination number whose soft count disagrees with the integer one.

### Commands

| command | purpose |
| --- | --- |
| `mizuho run cfg.yaml` | predict structures; `--top N` writes N structures of distinct bond-graph topology classes |
| `mizuho check cfg.yaml ref.cif` | evaluate the objective at a reference structure: per-term breakdown, integer and soft coordination numbers, detected space group. No optimisation |
| `mizuho fit cfg.yaml ref.cif` | fix the reference's Wyckoff assignment, optimise lattice and coordinates, and compare with the reference (lattice deviation, bound contacts, impostor warning). `--budget 4` is the global-minimum gate's budget |
| `mizuho candidates cfg.yaml` | run Phase 1 only and list its ranking after merging equivalent assignments, with free-parameter strata; `--seeds 5` reports the rank spread, `--find 'Mg@a Al@d O@e'` locates one assignment |
| `mizuho wyckoff SG` | list a space group's Wyckoff positions in mizuho's own setting, with the letters its Euclidean normalizer makes equivalent |
| `mizuho compare cfg.yaml ref.cif cand...` | diagnostic: put each candidate (CIF, or an impostor JSON record) in a topology class by its periodic bond graph, and against the reference list cell strain (volume and shape), bond-length, bond-angle and polyhedral-distortion shifts per topological atom type, and contacts the config's own cutoffs disagree about ([ADR-0018](docs/adr/0018-bond-graph-diagnostic.md)) |
| `mizuho suggest ref.cif` | reserved |

`check` and `fit` read the CIF in any setting or origin choice and convert
it to mizuho's (the first Hall setting of each space group), idealising the
rounded special positions published CIFs contain. See
[ADR-0013](docs/adr/0013-cli-subcommands.md); the former `mizuho cfg.yaml`
form is gone.

## How it works

Two execution modes. Mode A runs when `spacegroup` is given. Mode B is
experimental and opt-in: it runs only with `mode: auto_spacegroup` and
`spacegroup` omitted, because its known-answer tests do not pass yet (NaCl is
detected as P4mm, 99, instead of Fm-3m, 225;
[ADR-0021](docs/adr/0021-auto-spacegroup-search-stays-default-only-if-it-reaches-known-answers.md)).
A config with neither is rejected. `mode: single_spacegroup` may spell out
mode A; `mode` must agree with whether `spacegroup` is present:

1. **Mode A -- single space group** (`spacegroup: <1-230>`): run the
   Wyckoff-assignment search and numerical optimisation directly against
   the given space group.
2. **Mode B -- automatic space-group search, experimental**
   (`mode: auto_spacegroup`, `spacegroup` omitted): a
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
   screening instead, or `exhaustive` to run every one of the 230 groups
   through mode A's pipeline (hours for a binary compound;
   `spacegroup_search.spacegroups` restricts it to a subset). Every
   strategy ranks the entries that satisfy the constraints by symmetry
   first, so a structure is reported in its highest group rather than in a
   subgroup that also reaches zero error
   ([ADR-0026](docs/adr/0026-exhaustive-spacegroup-strategy-and-symmetry-first-ranking.md)).

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
| `spacegroup` | Space-group number (1-230); required unless `mode: auto_spacegroup` |
| `mode` | `auto_spacegroup` opts in to the experimental automatic search (mode B, `spacegroup` omitted); `single_spacegroup` is an optional spelling of mode A |
| `ionic_charges` | `{element: charge}` for ionic-mode searches |
| `lattice_bounds` | Per-axis `[min, max]` bounds for lattice constants |
| `bond_constraints` | List of `{atom1, atom2, distance_min, distance_max, weight}` |
| `angle_constraints` | List of `{atom_center, atom_neighbor, angle_min, angle_max, weight, exclude_neighbor_pairs}` |
| `coordination_constraints` | List of `{center_atom, neighbor_atom, cn_min, cn_max, cutoff_distance}` |
| `tree_search` | Wyckoff-search budget: `max_sites`, `max_formula_units`, `max_candidates`; `strategy: exhaustive` for the DFS oracle |
| `spacegroup_search` | Mode-B tuning: `top_sg_count`, `mcts_budget`, `confirm_after_visits`, `crystal_system_hint`; `strategy: flat` for the original 3-stage screening, `strategy: exhaustive` (optional `spacegroups`, and `record: file.jsonl` to record each group and resume) for all 230 groups |
| `optimization` | CMA-ES / L-BFGS-B settings: `n_restarts`, `cma_maxiter`, `seed`, ...; `topology_grouping` (see below) |
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

### Constraints derived from cited coordination numbers

For a compound with no reference structure, state the composition, `ionic_charges` and, per cation, its cited coordination number; mizuho derives the bond bands, cutoffs, coordination, angle and bond-valence constraints from them ([ADR-0028](docs/adr/0028-constraints-derived-from-cited-coordination-numbers.md)). A constraint the config states for the same pair wins.

```yaml
coordination_numbers: {Al: 6, Si: 4}
bonded_anions: {Si: [O]}     # optional: Si bonds O only (topaz); default: every anion
```

| Key | Default | Purpose |
|---|---|---|
| `coordination_numbers` | none | Cited CN per cation, an integer or `[min, max]`; turns derivation on |
| `bonded_anions` | every anion | Per cation, the anions it bonds (a cited fact); the others are held outside its shell |
| `auto_hydroxyl` | `true` | With H: at most one H per anion; `false` for hydrates |
| `auto_hydrogen_bond` | `true` | With H at CN 1: an anion within 2.2 A of H sits at 130 deg or more from the O-H bond |
| `optimization.centring` | `true` | After Stage 2, move a zero-error point to where its derived bonds are nearest their ideal lengths (bond valence V / CN), within 1e-4 of its error and in the same topology class; pulls a cell strained inside the wide derived bands back |

### Automatic anion coupling

Whenever `ionic_charges` is set and some cation-centred `coordination_constraint`/`angle_constraint` references an anion as its neighbour (directly, via `X`, or via an explicit list) but that anion has no constraint of its own as a center, mizuho auto-injects a default anion-centred pair (`neighbor_atom`/`atom_neighbor: X`) so the cation polyhedra sharing that anion are coupled to a mutually consistent geometry instead of drifting apart independently. Tuning/opt-out keys:

| Key | Default | Purpose |
|---|---|---|
| `auto_anion_coupling` | `true` | Disable the whole feature |
| `auto_anion_coupling_angle` | `true` | Keep the coordination constraint but skip the angle one |
| `auto_coupling_cn_range` | `[2, 4]` | Override the injected `cn_min`/`cn_max` |
| `auto_coupling_angle_range` | `[80, 160]` | Override the injected `angle_min`/`angle_max` |
| `auto_coupling_weight` | `1.0` | Override the injected weight |
| `auto_coupling_exclude_hh` | `true` | Only when H is the sole cation: auto-add `exclude_neighbor_pairs: [[H, H]]` to the injected angle constraint. Whenever other cations are present, the injected constraints name those cations and leave H out entirely ([ADR-0028](docs/adr/0028-constraints-derived-from-cited-coordination-numbers.md)) |

### Penalty normalisation

Every penalty term divides its deviation by a natural physical scale for that term before squaring: the target band's own midpoint for bond, angle, coordination-number (floored at half a neighbour, so a CN band of [0, 0] stays finite), volume, aspect-ratio and cell-validity terms; `distance_min` for repulsion; the cell's total absolute ionic charge for charge balance. All terms are therefore comparable relative deviations and each constraint's `weight` means what it says. Without this, the coordination-number term, a raw count, made up 65-99% of the total error across this project's own fixtures regardless of the nominal weights. The overlap term is already relative by construction; `hybridisation_constraints` stay in raw degrees.

Normalisation is always on. The former `normalize_penalties` key was removed and a config that still sets it is rejected with an error ([ADR-0009](docs/adr/0009-always-normalise-penalties.md)): the gates had been scoring the raw objective while every search optimised the normalised one.

### Topology classes of the search's end points

Every Stage-1 end point is put in a bond-graph topology class (coordination sequences and
polyhedral linkage, `mizuho/bond_graph.py`, with bonds taken from the config's own cutoffs). A
class is one structure for a DFT starting point: DFT relaxed every recorded impostor of the
published structure's class to the published structure
([ADR-0019](docs/adr/0019-topology-class-is-the-unit-of-success.md)). So:

- an assignment's restarts stop once two end points at or below `zero_error_threshold` share a
  class (every backend; MCMC and Optuna classify only points at or below the threshold);
- `run --top N` returns N structures of different classes, and the global ranking shows one row
  per class ("Same" counts the merged rows).

```yaml
optimization:
  topology_grouping:
    zero_error_threshold: 1.0e-3   # end points at or below this satisfy the constraints
    early_stop: true               # false runs every restart (e.g. to re-measure a budget ladder)
    early_stop_hits: 2
```

A config without any `cutoff_distance` or bond `cutoff` has no bonds, and so no classes:
grouping and early stopping are then off.

### Bond-valence sums

When `ionic_charges` and `bond_constraints` are given, each cation's bond-valence sum
(Brese & O'Keeffe 1991 R0, b = 0.37 A; `mizuho/data/bond_valence.py`, all 966 tabulated
pairs) is held to its formal valence within 0.1 valence units. Bonds are the pairs of each
bond constraint within its `cutoff`. Anion sums are reported, not added. Override per
element only where a reference structure itself exceeds 0.1, and say so beside the value:

```yaml
bond_valence:
  weight: 1.0              # default
  tolerance: {Ti: 0.28}    # measured at the reference (default 0.1)
```

`bond_valence: false` switches the term off. See
[ADR-0016](docs/adr/0016-bond-valence-sums.md).

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

The search gates of [ADR-0005](docs/adr/0005-five-gates-for-the-search.md)
(rank over five seeds, global minimum at x4 budget, end-to-end proxy) and the
automatic space-group known-answer tests take minutes to hours per mineral and
run only on request:

```bash
pytest --run-gates tests/integration/test_search_gates.py
pytest --run-gates tests/integration/test_auto_spacegroup_known_answers.py
```

`MIZUHO_GATE_RESULTS` names the directory the gates write their measurements
to (default `gate-results/`); `scripts/record_impostor.py` turns a
global-minimum result into an impostor fixture for the fast regression gate.

## License

MIT -- see [LICENSE](LICENSE).
