# One batched objective kernel for CPU and GPU, and a lockstep scheduler to feed it

An exhaustive composition-free run (`runs/alon`, ADR-0033) took 29.5 h on six Apple M4 workers
for 127 of 230 space groups (SG 166 alone 2.3 h). The target machine for longer runs is the
Kyushu University supercomputer Genkai, node group B. Design review of 2026-10-28.

## Measured

- **The objective is Python-bound, not arithmetic-bound.** Topaz (Pnma, 36 atoms, 18
  parameters): `compute_error` 13.0 ms per evaluation on one M4 core. The distance arithmetic
  (36 x 36 x 27 images) is microseconds. Of 82 s for one reduced-budget candidate
  (`optimize_one_assignment`, n_restarts 2, cma_maxiter 60, refine_maxiter 50), 98 % is
  `compute_error`: orbit deduplication (`expand_orbit`'s `any()` generator) 12 s,
  `all_image_distances` 13 s, soft CN 10 s, `np.triu_indices` 8 s, the function's own loops
  15 s.
- **By stage:** coordination-shell seed (local-shell CMA-ES) 32 s, Stage 1 CMA-ES 19 s,
  Stage 2 L-BFGS-B 31 s, of which 29 s is the finite-difference gradient.
- **CMA-ES evaluates its population one point at a time** (`[objective(x) for x in xs]`);
  pycma's default population at 18 dimensions is about 12.
- **Genkai node group B** (cc.kyushu-u.ac.jp, hardware, resource-group and fee pages, read
  2026-10-28): 4 x NVIDIA H100 (94 GiB, FP64 33.5 TFLOPS each, half the FP32 rate) and
  2 x Xeon Platinum 8490H (120 cores) per node. `b-batch` allocates 30 cores and 226.7 GiB per
  GPU, at most 168 h. Group B costs 120 pt per node-hour, so **one GPU slot costs 30 pt/h, the
  same as one whole CPU-only node of group A (120 cores)**. CUDA module 12.2.2, driver
  615.71.09, system Python 3.6.8, Singularity 4.1.3; JAX is not provided.
- **JAX 0.11.0 requires Python >= 3.12**; mizuho declared >= 3.10.

A single evaluation sent to a GPU would be slower than on a CPU (launch latency). A GPU pays off
only for large batches compiled into few fused kernels. The same rewrite removes the Python
overhead on a CPU too.

## Decision

### Goal and acceptance

- **The goal is throughput of exhaustive searches at today's scale** (primitive cell <= ~21
  atoms). Larger cells would be a separate decision.
- **Go/no-go is cost-normalised against Genkai group A.** One GPU slot (H100 + 30 cores,
  30 pt/h) must finish the benchmark (see Verification) at least **2x faster** than one group-A
  node (120 cores, 30 pt/h) running the new CPU kernel. That equals 8x over the same slot's 30
  cores. A no-go stops only the GPU-specific work (phase P4 of the plan); the kernel and the
  scheduler stay, because they speed up CPU runs.
- **GPU support is an implementation detail.** Algorithms, budgets, populations, restart
  counts and defaults do not change. Changing any of them because a GPU makes it cheap needs its
  own ADR measured on the ADR-0005 gates.

### Agreement between CPU and GPU

Bit-for-bit agreement across devices is not possible: summation order differs, and a CMA-ES
trajectory diverges on the last bit. The contract has three parts.

1. **Per evaluation:** total and per-term values agree with the NumPy reference within
   |Δf| <= 1e-9 · max(1, f).
2. **Run-to-run on one GPU:** the same configuration, seed and bucket layout give bit-identical
   results (XLA deterministic ops).
3. **Per search:** the ADR-0005 gates reach the same topology class (ADR-0019) and the same
   detected space group (ADR-0010) as on the CPU. Error values themselves are not compared.

The CPU path keeps its current bit-reproducibility.

### Architecture

- **One kernel, used at run time on CPU and GPU.** For each Wyckoff assignment, a *static
  plan* is resolved once: orbit operation indices, pair and species indices, and constraint
  tables. A batched array kernel then evaluates many parameter vectors against that plan.
  `objective/energy.py` is kept as the **reference oracle for tests**, not on the run-time
  path. Writing a fourth mirror (beside `energy.py`, `breakdown.py` and `energy_jax.py`) for
  the GPU alone was rejected.
- **JAX**, float64 throughout (`jax_enable_x64`). There is no mixed precision, and FP64 Tensor
  Cores are not targeted, since the work is not matrix products. PyTorch eager has a high
  per-op cost on small arrays on the CPU. CuPy has no fusion, no autodiff and no CPU path.
  Numba-CUDA splits the CPU and GPU code and has no autodiff.
- **JAX becomes a required dependency and `requires-python` rises to `>= 3.12`.** Python 3.10
  and 3.11 are dropped. GPU support comes from an extra, `mizuho[cuda12]`, which installs
  `jax[cuda12]`. Keeping `energy.py` as a run-time fallback without JAX would mean two run-time
  implementations.
- **Value-dependent decisions are reproduced exactly with masks over static shapes.** None is
  approximated.
  - **Orbit deduplication** (tolerance 1e-4, which does change the atom count near a special
    position) uses the generic-point operation list of each letter, plus a sequential "drop
    if an earlier kept point is within tolerance" mask.
  - **Angle and CN neighbour lists** have a static capacity K. **An overflow recompiles with a
    larger K and re-evaluates; nothing is truncated silently.**
  - Changing the reference's behaviour instead, e.g. to a fixed orbit without deduplication,
    was rejected under "GPU support is an implementation detail".
- **The void term is evaluated in two stages.** The grid bounds on R_hole are computed in the
  kernel. Only the rows the bounds cannot decide get the exact Voronoi value (qhull) on the host,
  which is filled back in. The term is active only without a density or volume constraint;
  `runs/alon` does not use it.

### Batching and processes

- **A lockstep scheduler, across candidates.**
  - The optimisers become ask/tell state machines: the Phase 1 local-shell solves, the
    coordination-shell seed, Stage 1, Stage 2 and centring.
  - One evaluator collects every pending request (candidates x restarts x population), groups
    them into **fixed-shape buckets**, and evaluates each bucket in one call.
  - The CPU path runs the same control flow.
  - A population alone (about 12 rows) or a restart wave (50-100 rows) is too small for a GPU.
    One process per worker sharing one GPU wastes it and contends for device memory.
- **Determinism under batching** holds when a row's value does not depend on the other rows in
  its batch or on its position in the batch. This is verified bitwise (see Verification). If
  it fails, the in-flight window is fixed deterministically instead.
- **Restart early stopping is unchanged.** The endpoint ledger already runs restarts in waves
  and drops results past the stop in restart order (ADR-0019), so batching a wave does not
  change the outcome.
- **Stage 2 keeps the finite-difference gradient by default.** The n+1 points perturbed by
  scipy's step rule are evaluated as one batch. `optimization.refine_jac` becomes an exact
  autodiff gradient of every term, replacing the CN-and-volume hybrid of phase I-2. Making it
  the default needs its own ADR measured on the gates.
- **Only the `cma` backend runs batched.** It is the default, and the local-shell solves use it
  too.
  - DE, basin hopping, dual annealing, MCMC and Optuna call the kernel one row at a time on the
    CPU. Their algorithms are unchanged; for example, scipy DE's `vectorized=True` would force
    deferred updating, so it is not used.
  - `--device gpu` with a non-`cma` backend is rejected with the reason: one row at a time is
    slower on a GPU.
- **One process owns one device and runs one scheduler.**
  - Within the process, a helper pool does the CPU-only work: Phase 1 enumeration of upcoming
    units, spglib and bond-graph classification, and the Voronoi fill-in. It has 29 workers on
    a GPU slot.
  - A window of (space group, composition) units is in flight, so the GPU never idles.
  - **Several GPUs, or the 120 cores of an A node, are used by sharding space groups across
    processes** (`--shard i/n`). Each shard writes its own record file. A final run reads every
    shard's record and only ranks and promotes (ADR-0010).
  - Driving four GPUs from one process with JAX sharding was rejected as more complex for no
    measured gain.
- **`restart_workers`, `candidate_workers` and `feasibility_workers` are removed.** A config that
    still sets one is rejected with a pointer here, as ADR-0009 and ADR-0033 did for theirs.
    `--workers` now sizes the helper pool.

### Interface and records

- **`--device {cpu,gpu}`** and YAML `compute: {device: ...}`, **default `cpu`, no
  auto-detection**: auto-detection would let results depend silently on the machine.
- `gpu` without CUDA JAX is an error with the install command.
- With `gpu`, mizuho sets `--xla_gpu_deterministic_ops=true` itself.
- The run header and the record state the device, the jax and jaxlib versions, and the driver
  version.

### Removed or rewritten

- `objective/energy_jax.py` is removed (superseded by the kernel's autodiff).
- The `numba` extra is removed. The remaining callers of `pairwise_distances_pbc` (bond graph,
  initial guess) are off the hot path.
- `breakdown.py` keeps its detailed NumPy report. The kernel returns per-term totals, and tests
  assert that the two agree.
- `scripts/benchmark_lattice.py` and `scripts/benchmark_jax_gradient.py` are removed or
  rewritten against the kernel.

### Out of scope

- Cells beyond today's scale.
- MIG slices: the fee page does not state how they are charged.
- Algorithm or budget changes that a GPU makes cheap.

## Verification (required before each phase is called done)

- **Oracle agreement** (part 1 of the contract):
  - every fixture in `tests/fixtures`, `impostors` and `auto`;
  - at the reference structure, random points within bounds, points within 2e-4 of a special
    position, configurations that overflow the neighbour capacity, and skewed triclinic cells;
  - total and per-term.
- **Gradients:**
  - the batched finite difference matches `scipy.optimize.approx_derivative`;
  - the autodiff gradient matches finite differences on a small synthetic case, with
    translation invariance checked as a first pass, not as proof.
- **GPU determinism:**
  - a repeated run is bit-identical;
  - a row's value is bit-identical whatever its position and its batch-mates.
- **Gates:** the ADR-0005 gates under `--device gpu` reach the CPU's topology class and
  detected space group.
- **Tests:** a `gpu` pytest marker runs only with `--run-gpu` and CUDA present. Everything else
  passes on JAX-CPU on a Mac.
- **Benchmark for the go/no-go:**
  - `runs/alon/alon.yaml` restricted to SG 166, 71 and 139 (the three slowest recorded), plus
    the topaz and adamite fixtures;
  - metric wall time x pt/h;
  - results and code state under `runs/gpu_benchmark/`.

The phases (P1 kernel, P2 H100 spike, P3 scheduler, P4 Genkai deployment and final go/no-go) are
in [plan-gpu-batched-evaluation.md](../plan-gpu-batched-evaluation.md).
