//! Run counters ([`Stats`]), early stopping ([`Budget`], [`Limits`]) and why a run ended
//! ([`Stop`]).
//!
//! Every algorithm asks its budget **before every objective evaluation**, with the counters so
//! far, so a run stopped by [`Limits::max_evaluations`] `= n` has evaluated exactly `n`
//! genomes.

pub use search_core::Budget;

/// Counters collected during one run, handed to the [`Budget`] before every evaluation and
/// returned in every outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Objective evaluations performed so far, including those of local searches.
    pub evaluations: u64,
    /// Completed generations (or iterations of a non-generational algorithm such as RMHC,
    /// where one iteration is one evaluation).
    pub generations: u64,
    /// The cost of the best genome evaluated so far; `+∞` before the first evaluation.
    pub best_cost: f64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            evaluations: 0,
            generations: 0,
            best_cost: f64::INFINITY,
        }
    }
}

/// Simple limits on evaluations, generations and the target cost.
///
/// [`Limits::DEFAULT`] allows `10^6` evaluations, which keeps every run finite. Any
/// `FnMut(&Stats) -> bool` is a [`Budget`] too, for example to add a wall-clock deadline in a
/// `std` crate (annotate the argument type: `|stats: &Stats| ...`).
///
/// ```
/// use evolutionary_planning::{Budget, Limits, Stats};
///
/// let mut limits = Limits { target_cost: Some(1e-9), ..Limits::evaluations(500) };
/// let stats = Stats { evaluations: 500, ..Stats::default() };
/// assert!(limits.exhausted(&stats));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Stop once this many evaluations have been performed. `None` is unbounded.
    pub max_evaluations: Option<u64>,
    /// Stop once this many generations have completed. `None` is unbounded.
    pub max_generations: Option<u64>,
    /// Stop once the best cost is at or below this value. `None` never stops on quality.
    pub target_cost: Option<f64>,
}

impl Limits {
    /// `10^6` evaluations, no other limit.
    pub const DEFAULT: Self = Self {
        max_evaluations: Some(1_000_000),
        max_generations: None,
        target_cost: None,
    };

    /// No limits at all: the run ends only through an algorithm's own convergence test.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_evaluations: None,
            max_generations: None,
            target_cost: None,
        }
    }

    /// Stops after `max` evaluations, with no other limit.
    #[must_use]
    pub const fn evaluations(max: u64) -> Self {
        Self {
            max_evaluations: Some(max),
            ..Self::unlimited()
        }
    }

    /// Stops after `max` generations, with no other limit.
    #[must_use]
    pub const fn generations(max: u64) -> Self {
        Self {
            max_generations: Some(max),
            ..Self::unlimited()
        }
    }

    /// Adds a target cost to these limits.
    #[must_use]
    pub const fn with_target(self, target: f64) -> Self {
        Self {
            target_cost: Some(target),
            ..self
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Budget<Stats> for Limits {
    fn exhausted(&mut self, stats: &Stats) -> bool {
        self.max_evaluations
            .is_some_and(|max| stats.evaluations >= max)
            || self
                .max_generations
                .is_some_and(|max| stats.generations >= max)
            || self
                .target_cost
                .is_some_and(|target| stats.best_cost <= target)
    }
}

/// Why a run ended.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The [`Budget`] said stop.
    Budget,
    /// An algorithm-specific stagnation criterion fired (for example RMHC's restart limit,
    /// the diffusion model's stagnation counters, or DE's population collapse).
    Stagnation,
}
