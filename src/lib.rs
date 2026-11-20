//! # Evolutionary planning
//!
//! Evolutionary planning: genetic algorithms, evolution strategies and multi-objective evolutionary search over plans, paths and trajectories.
//!
//! ## Planned scope
//!
//! - Genetic algorithms over action sequences, with plan repair and validity-aware operators
//! - Evolution strategies (e.g. CMA-ES) for continuous trajectories and controls
//! - Multi-objective evolutionary search: NSGA-II/III, MOEA/D, Pareto archives
//! - Quality diversity (MAP-Elites) for diverse plans
//! - Memetic hybrids with the heuristic and sampling planners of this ecosystem
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is not started yet and will change.
//!
//! Repository: <https://github.com/sunsided/evolutionary-planning>

#![no_std]

/// How good a candidate solution is; larger is better.
///
/// Fitness is evaluated by the caller, so plans, paths and trajectories can
/// all be scored by their own simulator or validator.
pub trait Fitness<G> {
    /// The fitness value, for example negated plan cost.
    type Value: PartialOrd;

    /// Evaluates `genome`.
    fn fitness(&mut self, genome: &G) -> Self::Value;
}
