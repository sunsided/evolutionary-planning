//! Integration tests: same-seed-same-result for every algorithm, evolutionary search under
//! an STL constraint, GLEAM action chains evolved end to end, and property tests for the
//! Pareto utilities and range invariants.

mod chains;
mod determinism;
mod pareto;
mod ranges;
mod stl_constrained;
