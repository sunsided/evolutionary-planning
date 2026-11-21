//! Integration tests: same-seed-same-result for every algorithm, evolutionary search under
//! an STL constraint, GLEAM action chains evolved end to end, and Pareto property tests.

mod chains;
mod determinism;
mod pareto;
mod stl_constrained;
