//! # Evolutionary planning
//!
//! Genetic algorithms, evolution strategies and multi-objective evolutionary search over
//! plans, paths and trajectories, for `no_std` + `alloc`.
//!
//! Every objective is **minimised**; costs are `f64` and NaN counts as `+∞`. All randomness
//! comes from a caller-provided [`rand_core::Rng`] (the crate ships [`SplitMix64`]), so a
//! seed reproduces a run. Every run takes a [`Budget`] that is asked before each evaluation;
//! [`Limits`] covers evaluations, generations and a target cost.
//!
//! See `docs/design.md` in the repository for the architecture, the paper table and the
//! design rules.
//!
//! Repository: <https://github.com/sunsided/evolutionary-planning>

#![no_std]

extern crate alloc;

pub mod budget;
pub mod cmaes;
pub mod cost;
pub mod es;
mod evaluator;
pub mod ga;
pub mod hill_climb;
pub mod objective;
pub mod observer;
pub mod operators;
pub mod problems;
pub mod rng;

pub use budget::{Budget, Limits, Stats, Stop};
pub use cost::{Individual, ranking, sanitize};
pub use evaluator::Outcome;
pub use objective::{MultiObjective, MultiObjectiveFn, Objective, multi_objective_fn};
pub use observer::{Event, Observer, ObserverFn, observer_fn};
pub use rng::{SplitMix64, below, chance, standard_normal, uniform, unit_f64};
