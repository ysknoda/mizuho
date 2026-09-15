# The periodic-image change is unconditional, with no compatibility flag

This project's convention for behaviour-changing work is an opt-in config key defaulting to the old
behaviour (`normalize_penalties`, `optimization.refine_jac`). ADR-0001 deliberately breaks that
convention and replaces the old counting outright.

Those earlier flags guard **design choices**, where both settings are defensible and the user picks
one. Minimum-image counting is not a design choice: it is an implementation that cannot express
NaCl's CN of 6. Keeping it behind a flag would mean maintaining, testing and documenting a mode
that is known to be wrong, and every fixture and target would have to be validated twice.
