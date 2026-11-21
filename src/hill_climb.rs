//! Random-mutation hill climbing (RMHC): mutate a copy of the current genome and keep it if
//! it is not worse.
//!
//! Accepting equal costs lets the climber drift across plateaus (Olesen et al. 2020 accept
//! "better than or equal"); optional restarts from a fresh random genome escape local optima
//! after a stagnation count.
//!
//! ```
//! use evolutionary_planning::hill_climb::{Config, HillClimber};
//! use evolutionary_planning::operators::discrete::{BitFlip, RandomBits};
//! use evolutionary_planning::problems::single::one_max;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut climber = HillClimber::new(Config::default(), RandomBits { len: 30 }, BitFlip::default());
//! let mut objective = |bits: &Vec<bool>| one_max(bits);
//! let outcome = climber.run(&mut objective, &mut Limits::evaluations(5_000).with_target(0.0), &mut (), &mut SplitMix64::new(1));
//! assert_eq!(outcome.best_cost(), 0.0);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::evaluator::Evaluator;
use crate::operators::{Initializer, Mutation};
use crate::{Budget, Event, Individual, Objective, Observer, Outcome, Stats, Stop};

/// Parameters of a [`HillClimber`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Accept a mutant of equal cost (plateau moves). Default `true`.
    pub accept_equal: bool,
    /// Restart from a fresh random genome after this many consecutive non-improving
    /// iterations; `None` never restarts. The best genome over all restarts is returned.
    pub restart_after: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            accept_equal: true,
            restart_after: None,
        }
    }
}

/// Random-mutation hill climbing over any genome.
#[derive(Debug, Clone, PartialEq)]
pub struct HillClimber<I, M> {
    /// The run parameters.
    pub config: Config,
    /// Creates the start genome and restart genomes.
    pub init: I,
    /// Produces the neighbour of the current genome.
    pub mutation: M,
}

impl<I, M> HillClimber<I, M> {
    /// Creates a hill climber.
    #[must_use]
    pub fn new(config: Config, init: I, mutation: M) -> Self {
        Self {
            config,
            init,
            mutation,
        }
    }

    /// Climbs from a random genome until the budget stops it. One iteration (one evaluation)
    /// counts as one generation.
    pub fn run<G, O, B, Obs, R>(
        &mut self,
        objective: &mut O,
        budget: &mut B,
        observer: &mut Obs,
        rng: &mut R,
    ) -> Outcome<G>
    where
        G: Clone,
        I: Initializer<G>,
        M: Mutation<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let start = self.init.initialize(rng);

        self.run_from(start, objective, budget, observer, rng)
    }

    /// Climbs from `start` until the budget stops it.
    pub fn run_from<G, O, B, Obs, R>(
        &mut self,
        start: G,
        objective: &mut O,
        budget: &mut B,
        observer: &mut Obs,
        rng: &mut R,
    ) -> Outcome<G>
    where
        G: Clone,
        I: Initializer<G>,
        M: Mutation<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let mut evaluator = Evaluator::new(objective, budget, observer);
        let Some(mut current) = evaluator.individual(start) else {
            return evaluator.outcome(Vec::new(), Stop::Budget);
        };

        let mut stale = 0;
        let mut restarts = 0;
        loop {
            if self
                .config
                .restart_after
                .is_some_and(|limit| stale >= limit)
            {
                restarts += 1;
                evaluator.emit(Event::Restart {
                    restart: restarts,
                    population: 1,
                });
                let Some(fresh) = evaluator.individual(self.init.initialize(rng)) else {
                    break;
                };
                current = fresh;
                stale = 0;
                continue;
            }

            let Some(improved) = self.step(&mut current, &mut evaluator, rng) else {
                break;
            };
            stale = if improved { 0 } else { stale + 1 };
            evaluator.end_generation(current.cost);
        }

        evaluator.outcome(alloc::vec![current], Stop::Budget)
    }

    /// One iteration: mutate, evaluate, accept. Returns whether the cost strictly improved,
    /// or `None` if the budget is exhausted.
    pub(crate) fn step<G, O, B, Obs, R>(
        &mut self,
        current: &mut Individual<G>,
        evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        rng: &mut R,
    ) -> Option<bool>
    where
        G: Clone,
        M: Mutation<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let mut candidate = current.genome.clone();
        self.mutation.mutate(&mut candidate, rng);
        let cost = evaluator.evaluate(&candidate)?;
        let improved = cost < current.cost;
        if improved || (self.config.accept_equal && cost <= current.cost) {
            *current = Individual {
                genome: candidate,
                cost,
            };
        }

        Some(improved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::discrete::{BitFlip, RandomBits};
    use crate::problems::single::leading_ones;
    use crate::{Limits, SplitMix64};

    /// RMHC solves LeadingOnes, exactly within budget and deterministically.
    #[test]
    fn solves_leading_ones() {
        let run = || {
            let mut climber = HillClimber::new(
                Config::default(),
                RandomBits { len: 20 },
                BitFlip::default(),
            );
            let mut objective = |b: &Vec<bool>| leading_ones(b);
            climber.run(
                &mut objective,
                &mut Limits::evaluations(20_000).with_target(0.0),
                &mut (),
                &mut SplitMix64::new(2),
            )
        };
        let outcome = run();

        assert_eq!(outcome.best_cost(), 0.0);
        assert_eq!(outcome, run());
    }

    /// Restarts are reported and the best over all restarts is kept.
    #[test]
    fn restarts_keep_best() {
        let mut restarts = 0;
        let mut climber = HillClimber::new(
            Config {
                restart_after: Some(10),
                ..Config::default()
            },
            RandomBits { len: 8 },
            BitFlip::default(),
        );
        // A deceptive trap: all ones is optimal, otherwise more zeros is better.
        let mut trap = |b: &Vec<bool>| {
            let ones = b.iter().filter(|&&x| x).count();
            if ones == 8 {
                0.0
            } else {
                1.0 + f64::from(u32::try_from(ones).expect("small"))
            }
        };
        let outcome = {
            let mut observer = crate::observer_fn(|e| {
                if let Event::Restart { .. } = e {
                    restarts += 1;
                }
            });
            climber.run(
                &mut trap,
                &mut Limits::evaluations(500),
                &mut observer,
                &mut SplitMix64::new(1),
            )
        };

        assert!(restarts > 0);
        assert_eq!(outcome.stats.evaluations, 500);
        assert!(outcome.best_cost() <= 1.0);
    }
}
