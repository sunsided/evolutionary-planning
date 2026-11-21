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
//! # Modules
//!
//! | Family | Modules |
//! |---|---|
//! | Core | [`operators`] (traits and representations), [`budget`], [`observer`], [`rng`], [`cost`], [`objective`], [`problems`] |
//! | Single objective | [`ga`], [`diffusion`], [`hill_climb`], [`es`], [`cmaes`], [`de`], [`memetic`] |
//! | Multi objective | [`pareto`], [`nsga2`], [`nsga3`], [`moead`] |
//! | Quality diversity | [`map_elites`] |
//! | Planning | [`rhea`], [`stl`], [`trajectory`], [`graph_path`], `sas` (feature `sas-plus`), [`programmer`], [`gp`] |
//!
//! ```mermaid
//! graph TD
//!     core[rng, cost, budget, observer] --> ops[operators]
//!     ops --> single[ga, diffusion, hill_climb, es, cmaes, de]
//!     single --> memetic
//!     ops --> multi[pareto, nsga2, nsga3, moead]
//!     ops --> qd[map_elites]
//!     single --> planning[rhea, stl, trajectory, graph_path, sas, programmer, gp]
//! ```
//!
//! # Example
//!
//! ```
//! use evolutionary_planning::cmaes::{CmaEs, Config};
//! use evolutionary_planning::problems::single::ellipsoid;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut cma = CmaEs::new(vec![1.0; 6], 0.5, Config::default());
//! let mut objective = |x: &Vec<f64>| ellipsoid(x);
//! let outcome = cma.minimize(&mut objective, &mut Limits::evaluations(10_000).with_target(1e-10), &mut (), &mut SplitMix64::new(7));
//! assert!(outcome.best_cost() <= 1e-10);
//! ```
//!
//! See `docs/design.md` in the repository for the architecture, the paper table, the design
//! rules and the deviations from the source papers.
//!
//! Repository: <https://github.com/sunsided/evolutionary-planning>

#![no_std]

extern crate alloc;

pub mod budget;
pub mod cmaes;
pub mod cost;
pub mod de;
pub mod diffusion;
pub mod es;
mod evaluator;
pub mod ga;
pub mod gp;
pub mod graph_path;
pub mod hill_climb;
pub mod map_elites;
pub mod memetic;
pub mod moead;
mod multi;
pub mod nsga2;
pub mod nsga3;
pub mod objective;
pub mod observer;
pub mod operators;
pub mod pareto;
pub mod problems;
pub mod programmer;
pub mod rhea;
pub mod rng;
#[cfg(feature = "sas-plus")]
pub mod sas;
pub mod stl;
pub mod trajectory;

pub use budget::{Budget, Limits, Stats, Stop};
pub use cost::{Individual, ranking, sanitize};
pub use evaluator::Outcome;
pub use multi::MultiOutcome;
pub use objective::{MultiObjective, MultiObjectiveFn, Objective, multi_objective_fn};
pub use observer::{Event, Observer, ObserverFn, observer_fn};
pub use rng::{SplitMix64, below, chance, standard_normal, uniform, unit_f64};
