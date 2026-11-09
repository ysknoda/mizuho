# Coding plan — batched objective kernel, lockstep scheduler, GPU evaluation

Decisions behind this plan:
[ADR-0034](adr/0034-batched-objective-kernel-and-gpu-evaluation.md). Every phase ends with the
full test suite green and the ADR-0034 verification items that apply to it.

## P1 — static plan and batched kernel (CPU)

Pays off on its own, on the Mac and on Genkai group A.

- Raise `requires-python` to `>=3.12`, make `jax` a required dependency, and add the
  `cuda12` extra (`jax[cuda12]`). Update the classifiers and the README install section.
- **Static plan** per Wyckoff assignment:
  - the generic-point orbit operation list per letter;
  - pair, image and species indices;
  - the expanded wildcard constraint tables;
  - neighbour capacities K.
- **Batched kernel** `(plan, P[B, n]) -> (total[B], per_term[B, T], flags[B])`:
  - every penalty term of `energy.py`;
  - orbit-deduplication mask;
  - capacity-overflow flag, which triggers a recompile with a larger K and a re-evaluation;
  - R_hole grid bounds with an "undecided" flag, filled in on the host by `cell_terms.r_hole`.
- Switch every run-time caller of `compute_error` to the kernel (single-row for non-`cma`
  backends). Keep `energy.py` as the oracle.
- `refine_jac` becomes autodiff of the whole kernel. Remove `objective/energy_jax.py` and the
  `numba` extra.
- Tests:
  - oracle agreement (all fixtures, special-position and overflow points);
  - per-term agreement with `breakdown.py`;
  - autodiff against finite differences.

## P2 — spike on a Genkai H100 (go/no-go for P4)

- Install miniforge with Python 3.12 or later and pinned `jax[cuda12]` on a login node.
  Record the versions.
- Measure the kernel's throughput (evaluations/s) against batch size (64 ... 16k) and atom
  count, for the benchmark assignments.
- Measure the same on one group-A node across processes x XLA threads.
- Estimate the batch sizes the scheduler would reach from `runs/alon/alon.jsonl` (candidates
  per unit x restarts x population).
- **Go if the projected cost-normalised speedup is >= 2x.** On no-go, P3 still proceeds for
  the CPU; P4 does not.

## P3 — lockstep scheduler

- Make the optimisers ask/tell state machines, in this order:
  - pycma's `ask`/`tell`;
  - the Phase 1 local-shell solves (`precompute_letter_feasibility`,
    `rank_by_feasibility_staged`);
  - the coordination-shell seed, Stage 1, Stage 2's batched finite difference (scipy's
    step rule) and centring.
- The evaluator collects requests into fixed-shape buckets, keyed by plan shape and padded to
  fixed batch sizes. It schedules deterministically (never in completion order).
- The endpoint ledger keeps its wave semantics.
- A helper process pool handles Phase 1 enumeration of upcoming units, classification and the
  Voronoi fill-in. A window of in-flight units keeps the device busy.
- Remove `restart_workers`, `candidate_workers` and `feasibility_workers` (rejected with a
  pointer to ADR-0034). `--workers` sizes the helper pool.
- `--device {cpu,gpu}`, `compute.device`, the deterministic XLA flag, device and version
  fields in the header and the record. Reject `--device gpu` with a non-`cma` backend.
- Tests:
  - results do not depend on the bucket layout (CPU, bitwise);
  - ledger early-stop equivalence;
  - rejection messages.

## P4 — Genkai deployment and final go/no-go

- `--shard i/n` with one record per shard, and a collecting run that ranks and promotes from
  all shard records.
- `scripts/genkai/` job-script templates:
  - one GPU slot;
  - a whole B node (4 shards);
  - a group-A node (CPU shards).
  The scheduler syntax is to be confirmed on the system.
- GPU tests (`gpu` marker, `--run-gpu`): bitwise repeatability and row-position invariance.
- The ADR-0005 gates under `--device gpu`: same topology class and detected space group as on
  the CPU.
- The benchmark (SG 166, 71 and 139 of `runs/alon/alon.yaml`, plus topaz and adamite) on one
  GPU slot against one group-A node, as wall time x pt/h, recorded under
  `runs/gpu_benchmark/`.
- Record the outcome in ADR-0034's follow-up: whether GPU is a supported path or documented as
  not recommended.
