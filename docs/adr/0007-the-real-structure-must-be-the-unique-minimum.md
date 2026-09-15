# The published structure must be the objective's *only* minimum

ADR-0003 requires the objective to be approximately zero at the published structure. That is
necessary and it is not sufficient: it never asked whether anything *else* is also zero. It is.

Handed the published topology and quadruple the production budget, the optimiser finds a point
**at least as good as the published structure**:

| fixture | free params | lattice deviation | cell volume | error found / at published, `normalize_penalties: true` | the same, `false` |
| --- | --- | --- | --- | --- | --- |
| diopside | 18 | −5.6 / +4.0 / +6.6 / +6.8 % | +0.2% | **0.00000000** / 0.00000066 | **0.00000000** / 0.00002282 |
| topaz | 18 | −13.6 / +14.3 / +14.0 % | +12.7% | **0.00000000** / 0.00000013 | **0.00000000** / 0.00000254 |
| adamite | 17 | +5.9 / +5.8 / −5.3 % | +6.1% | **0.00564958** / 0.00805674 | 0.17092684 / 0.02917244 |

Call such a point an **impostor**: a structure the objective rates at or below the published one
while its lattice is far outside the 3% target.

Diopside and topaz are impostors under **either** setting of `normalize_penalties`. Adamite's is one
only under `true` — under `false` the published structure wins by 6x. The probes that found these
were run with `true`, which is what every search run in this project has used (see
`docs/plan-periodic-image-counting.md`), while the fixtures set the key nowhere and so default to
`false`, which is what the ADR-0003 gate measures. **The gate and the search have been scoring
different objectives.** That is a defect in its own right, recorded below.

This reframes the whole problem. The optimiser is **not** the bottleneck — the budget ladder showed
18 dimensions are solvable (diopside: 0.019499 at x1, 0.000000 at x4). Spending more compute simply
arrives at the impostor faster and more reliably; at x1 diopside's lattice was off by
+2.1/+5.6/+6.6%, at x4 by −5.6/+4.0/+6.6/+6.8%. **Strengthening the optimiser made the prediction
worse.** An earlier decision to fix the optimiser first was withdrawn on this evidence.

## Two mechanisms, found by diagnosis

### 1. The soft coordination count can be gamed (diopside, adamite)

The coordination term uses a logistic soft count so the penalty has a usable gradient. That
smoothing is also a loophole: an atom parked *at* the cutoff contributes exactly one half.

Diopside's Ca, cutoff 3.095 A, constraint `cn_min: 7, cn_max: 8`:

```
published : 2.349 2.349 2.360 2.360 2.559 2.559 2.719 2.719 | 3.470 ...   hard 8, soft 7.999
impostor  : 2.301 2.301 2.443 2.443 2.850 2.850 3.096 3.096 | 3.327 ...   hard 6, soft 7.187
                                              ^^^^^ ^^^^^
```

Two oxygens sit 0.001 A outside the cutoff, each worth `expit(-0.01) = 0.4975`. Six real
neighbours plus two halves plus a tail passes `[7, 8]` with **zero penalty** while the integer
coordination is 6. Adamite does the same on Zn: soft 4.523 / 4.995 where the published structure
gives 5.005 / 5.982.

### 2. Bond bands are far wider than the chemistry they describe (topaz, diopside)

Topaz's impostor exploits no loophole at all — its coordination numbers are exactly integral. It
simply uses slack that was there all along:

| bond | span at the published structure | band as configured | ratio |
| --- | --- | --- | --- |
| Si–O | 1.636–1.642 (0.006 A) | [1.55, 1.75] (0.20 A) | **33x** |
| Al–O | 1.885–1.902 (0.017 A) | [1.80, 2.05] (0.25 A) | **15x** |
| Al–F | 1.790–1.801 (0.011 A) | [1.75, 1.95] (0.20 A) | **18x** |

Distorting every polyhedron to the band edges (Si–O 1.625–1.716, Al–O 1.808–2.026) buys a 12.7%
larger cell at no cost. ADR-0004's rule cannot catch this: it re-measures a band only when the
reference falls *outside* it, and here the reference sits comfortably inside.

## Decision

**A coordination number is an integer.** For every **cation-centred** coordination constraint,
`|CN_soft − round(CN_soft)|` must not exceed **0.05**. The form is the literal deviation with a
dead band, not a smooth surrogate such as `sin²(π·CN)`: the kink at integers lies inside the dead
band where the gradient is already zero, the kink at half-integers lies where the penalty is
maximal, and `dev = 0.1865` reads as "0.19 away from an integer" where `0.3076` reads as nothing.
The weight is **inherited from the parent coordination constraint** — integrality is a property of
that coordination number, not separate physics — so this introduces no new free parameter.

The 0.05 ceiling is chosen from data, not derived, and it separates two well-spaced groups: the
worst reference deviation across all four minerals is 0.0179 (adamite Zn), the mildest impostor
deviation is 0.1865 (diopside Ca). Roughly threefold clearance on each side.

**Anion-centred constraints are exempt.** A cation's coordination polyhedron is a discrete object;
the number of cations around an anion is not. Diopside's O deviates by 0.127 at its own published
structure even with the correctly measured cutoff, because the Ca coordination in a pyroxene is
genuinely broad. Setting the ceiling to 0.15 to accommodate that would be fitting the threshold to
the result.

**A bond band is the span measured at the reference structure, widened by 2% of the bond length**
(rounded outward, per ADR-0004's rounding rule). Measured against the three impostors, 2% rejects
all three topaz bonds and diopside's Mg–O and Ca–O; 3% lets topaz's Al–F through (impostor 1.850
against a band ceiling of 1.854). A fixed absolute margin was rejected: bond-length variation
scales with bond length, so 0.03 A is 1.8% at Si–O and 1.2% at Ca–O for no physical reason.

**The soft count is a numerical device, not the definition.** Gates and the production report judge
by the **integer** coordination number and flag any disagreement with the soft one. This changes no
landscape and costs no parameter; it is the net under the two fixes above.

Together the two fixes cover every known impostor — and they are complementary, neither sufficient
alone. Both criteria are purely geometric (soft coordination numbers and bond distances), so unlike
the error values above they do not depend on `normalize_penalties` at all:

| impostor | integrality ≤ 0.05 | band = span ± 2% |
| --- | --- | --- |
| diopside (either setting) | rejected (Ca 0.1865) | rejected (Mg–O, Ca–O) |
| topaz (either setting) | **passes** (0.0000) | rejected (all three bonds) |
| adamite (`normalize_penalties: true` only) | rejected (Zn 0.4773) | passes |

Diopside alone establishes that integrality is needed; topaz alone establishes that the band rule
is needed. Adamite corroborates integrality but cannot carry it on its own, since under the
fixtures' own setting it has no impostor to reject.

## What did not work

Anion-centred polymerisation was the leading hypothesis and it is **wrong**. Every fixture's
silicate polymerisation degree is uniquely fixed by its composition, its cation coordination
numbers and a per-atom band — diopside 8 Si x CN 4 = 32 Si–O bonds over 24 O in `[1, 2]` forces
exactly 16 non-bridging and 8 bridging, matching the measured `{1: 16, 2: 8}`; the same holds for
topaz `{1: 16}`, adamite `{0: 4, 1: 16}` and wollastonite `{1: 12, 2: 6}`. But **every impostor
reproduces its mineral's distribution exactly.** They are all correctly polymerised.

The constraint is still written into the fixtures, because it is a real chemical fact that no
fixture currently states and its effect on Phase 1 ranking is unmeasured — not because it restores
uniqueness. It does not.

## Consequences

- Deriving bond bands from the reference makes the reference gate's bond term **tautological**.
  This is acknowledged rather than hidden: the bond term is kept (removing it would break the
  identity `error_breakdown(...)["total"] == compute_error(...)`) and the ADR records that it
  carries no evidential weight there. Narrowing is the safe direction — it excludes impostors and
  can only wrongly reject a prediction that is slightly off, which is exactly what a 3% target
  asks for. Using a second, independent refinement per mineral as the gate's reference would remove
  the tautology entirely and is recorded as a future improvement; it needs a second published
  structure per mineral, which may not exist for adamite.
- Adamite's bands become very wide (Zn–O spans 0.415 A at its reference against topaz Si–O's
  0.006 A). That is left as measured. It states honestly that the 1937 data cannot constrain the
  bond length tightly — the same fact ADR-0003 already recorded by giving adamite a 5e-2 ceiling —
  and adamite's impostor is rejected on integrality regardless.
- **`normalize_penalties` must be settled before any of this is implemented.** Search runs use
  `true`, the fixtures and therefore the ADR-0003 gate use `false`, and the adamite result shows the
  two objectives genuinely disagree about which structure is better. Every threshold in this ADR was
  chosen against geometry rather than against error values precisely so that it survives the choice,
  but the gates cannot be trusted while the thing they measure differs from the thing being
  optimised. This is recorded as an open question, not a settled decision.
- Uniqueness is not provable. The global-minimum gate of ADR-0005 tests it where we can reach, the
  impostor regression pins the cases we found, and ADR-0005's two-round cap stops the hunt from
  running forever.
