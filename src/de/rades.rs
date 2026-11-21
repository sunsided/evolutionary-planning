//! RADES, rank-based differential evolution with a successful archive (Parque 2024,
//! equations (5)–(11)).
//!
//! Each target `i` keeps a stagnation counter `q_i`. While `q_i ≤ Q` the search exploits the
//! population: the reference vector is `x_i` and the mutant is
//! `x_best + F (x_r1 - x_r2)`. Once the target has failed more than `Q` times in a row it
//! explores from the archive of successful trials: reference `A_i`, mutant
//! `A_best + F (A_r1 - A_r2)`. The indices `r1`, `r2` are rank-biased by (9),
//! `r = ⌊|P| / (2β - 1) · (β - sqrt(β² - 4 (β - 1) u))⌋` into the cost-sorted population or
//! archive; `r2` is redrawn while it equals `r1` (the paper does not say, and an equal pair
//! gives a zero difference vector). The trial is `u = x^r + b ∘ (v - x^r)` with a binomial mask `b` (10)–(11), and it
//! replaces the target if strictly better (5); successful trials enter the archive, which is
//! capped at `|P|` by deleting a uniformly random member (the paper leaves the choice open).
//!
//! ```
//! use evolutionary_planning::de::rades::{Config, Rades};
//! use evolutionary_planning::operators::real::Bounds;
//! use evolutionary_planning::problems::single::sphere;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut rades = Rades::new(Config::default(), Bounds::uniform(4, -5.0, 5.0));
//! let mut objective = |x: &Vec<f64>| sphere(x);
//! let outcome = rades.run(&mut objective, &mut Limits::evaluations(20_000), &mut (), &mut SplitMix64::new(2));
//! assert!(outcome.best_cost() < 1e-6);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use super::{CrossoverKind, best_index, bounce_back, cross};
use crate::evaluator::Evaluator;
use crate::operators::Initializer;
use crate::operators::real::Bounds;
use crate::rng::{below, unit_f64};
use crate::{Budget, Individual, Objective, Observer, Outcome, Stats, Stop, ranking};

/// Parameters of a [`Rades`] run; the defaults are the paper's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The population size `NP`, at least `3`. Paper: `10`.
    pub population_size: usize,
    /// The scaling factor `F`. Paper: `0.7`.
    pub f: f64,
    /// The crossover probability `CR`. Paper: `0.5`.
    pub cr: f64,
    /// The stagnation threshold `Q`. Paper: `128`.
    pub stagnation: u64,
    /// The rank bias `β > 1` of (9); larger values bias less. Paper: `2`.
    pub beta: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: 10,
            f: 0.7,
            cr: 0.5,
            stagnation: 128,
            beta: 2.0,
        }
    }
}

/// The RADES optimiser over a box.
#[derive(Debug, Clone, PartialEq)]
pub struct Rades {
    /// The run parameters.
    pub config: Config,
    /// The search box; trials are bounced back into it as in [`De`](super::De).
    pub bounds: Bounds,
}

/// The evolving state of one run.
struct State {
    /// The population `P`.
    population: Vec<Individual<Vec<f64>>>,
    /// The archive `A` of successful trials, at most `|P|` long.
    archive: Vec<Individual<Vec<f64>>>,
    /// The stagnation counters `q_i`.
    stale: Vec<u64>,
}

impl Rades {
    /// Creates a RADES run.
    #[must_use]
    pub fn new(config: Config, bounds: Bounds) -> Self {
        Self { config, bounds }
    }

    /// Runs until the budget stops it.
    ///
    /// # Panics
    ///
    /// Panics if the population is smaller than `3` or `β <= 1`.
    pub fn run<O, B, Obs, R>(
        &mut self,
        objective: &mut O,
        budget: &mut B,
        observer: &mut Obs,
        rng: &mut R,
    ) -> Outcome<Vec<f64>>
    where
        O: Objective<Vec<f64>> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        assert!(
            self.config.population_size >= 3,
            "RADES needs a population of at least three"
        );
        assert!(self.config.beta > 1.0, "the rank bias must exceed one");

        let mut evaluator = Evaluator::new(objective, budget, observer);
        let mut state = State {
            population: Vec::with_capacity(self.config.population_size),
            archive: Vec::new(),
            stale: alloc::vec![0; self.config.population_size],
        };
        for _ in 0..self.config.population_size {
            let Some(individual) = evaluator.individual(self.bounds.clone().initialize(rng)) else {
                return evaluator.outcome(state.population, Stop::Budget);
            };
            state.population.push(individual);
        }

        // The paper initialises the archive from the population.
        state.archive.clone_from(&state.population);
        loop {
            if !self.generation(&mut state, &mut evaluator, rng) {
                return evaluator.outcome(state.population, Stop::Budget);
            }

            evaluator.end_generation(crate::ga::population_best(&state.population));
        }
    }

    /// One pass over all targets; `false` if the budget ran out.
    fn generation<O, B, Obs, R>(
        &self,
        state: &mut State,
        evaluator: &mut Evaluator<'_, Vec<f64>, O, B, Obs>,
        rng: &mut R,
    ) -> bool
    where
        O: Objective<Vec<f64>> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let population_order = ranking(&costs(&state.population));
        let archive_order = ranking(&costs(&state.archive));
        let best = best_index(&state.population);
        let archive_best = archive_order.first().copied();
        for i in 0..state.population.len() {
            let explore = state.stale[i] > self.config.stagnation && archive_best.is_some();
            let trial = if explore {
                let archive_i = &state.archive[i % state.archive.len()].genome;
                self.trial(
                    archive_i,
                    &state.archive,
                    &archive_order,
                    archive_best.unwrap_or(0),
                    rng,
                )
            } else {
                let x_i = state.population[i].genome.clone();
                self.trial(&x_i, &state.population, &population_order, best, rng)
            };

            let Some(cost) = evaluator.evaluate(&trial) else {
                return false;
            };

            if cost < state.population[i].cost {
                let individual = Individual {
                    genome: trial,
                    cost,
                };
                state.population[i] = individual.clone();
                Self::archive(state, individual, rng);
                state.stale[i] = 0;
            } else {
                state.stale[i] += 1;
            }
        }

        true
    }

    /// Builds the trial from reference `reference` and the rank-biased mutant over `source`.
    fn trial<R: Rng + ?Sized>(
        &self,
        reference: &[f64],
        source: &[Individual<Vec<f64>>],
        order: &[usize],
        best: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        let r1 = order[self.biased_rank(source.len(), rng)];
        let mut r2 = order[self.biased_rank(source.len(), rng)];
        // Equal indices give a zero difference vector; redraw (a bounded number of times).
        for _ in 0..16 {
            if r2 != r1 || source.len() < 2 {
                break;
            }
            r2 = order[self.biased_rank(source.len(), rng)];
        }
        let x = |k: usize| &source[k].genome;
        let mutant: Vec<f64> = (0..reference.len())
            .map(|d| x(best)[d] + self.config.f * (x(r1)[d] - x(r2)[d]))
            .collect();
        let mut trial = cross(
            reference,
            &mutant,
            self.config.cr,
            CrossoverKind::Binomial,
            rng,
        );
        bounce_back(&mut trial, reference, &self.bounds);

        trial
    }

    /// The rank-biased index (9) into a sorted list of `size` members.
    fn biased_rank<R: Rng + ?Sized>(&self, size: usize, rng: &mut R) -> usize {
        let beta = self.config.beta;
        let u = unit_f64(rng);
        #[expect(
            clippy::cast_precision_loss,
            reason = "population sizes are far below 2^52"
        )]
        let scale = size as f64 / (2.0 * beta - 1.0);
        let value = scale * (beta - libm::sqrt(beta * beta - 4.0 * (beta - 1.0) * u));

        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative rank below size"
        )]
        let rank = libm::floor(value) as usize;

        rank.min(size - 1)
    }

    /// Adds a successful trial to the archive, deleting a random member beyond `|P|`.
    fn archive<R: Rng + ?Sized>(state: &mut State, individual: Individual<Vec<f64>>, rng: &mut R) {
        state.archive.push(individual);
        if state.archive.len() > state.population.len() {
            let victim = below(rng, state.archive.len());
            state.archive.swap_remove(victim);
        }
    }
}

/// The costs of a population.
fn costs(population: &[Individual<Vec<f64>>]) -> Vec<f64> {
    population.iter().map(|i| i.cost).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::single::{rastrigin, rosenbrock};
    use crate::{Limits, SplitMix64};

    /// The rank bias stays in range and favours good ranks.
    #[test]
    fn biased_ranks() {
        let rades = Rades::new(Config::default(), Bounds::uniform(1, 0.0, 1.0));
        let mut rng = SplitMix64::new(1);
        let mut counts = [0usize; 10];
        for _ in 0..10_000 {
            counts[rades.biased_rank(10, &mut rng)] += 1;
        }

        assert!(counts[0] > counts[5]);
        assert_eq!(counts.iter().sum::<usize>(), 10_000);
    }

    /// RADES gets close to the Rosenbrock optimum and into the deep basins of Rastrigin. It
    /// is a greedy, best-based scheme tuned for small budgets (the paper uses 300
    /// evaluations), so it converges slowly in the curved valley and is not guaranteed to
    /// find Rastrigin's global optimum.
    #[test]
    fn optimises_standard_functions() {
        let mut rades = Rades::new(
            Config {
                population_size: 20,
                ..Config::default()
            },
            Bounds::uniform(2, -2.0, 2.0),
        );
        let mut objective = |x: &Vec<f64>| rosenbrock(x);
        let outcome = rades.run(
            &mut objective,
            &mut Limits::evaluations(20_000),
            &mut (),
            &mut SplitMix64::new(3),
        );
        assert!(outcome.best_cost() < 1e-2, "{}", outcome.best_cost());

        let mut rades = Rades::new(
            Config {
                population_size: 20,
                ..Config::default()
            },
            Bounds::uniform(3, -5.12, 5.12),
        );
        let mut objective = |x: &Vec<f64>| rastrigin(x);
        let outcome = rades.run(
            &mut objective,
            &mut Limits::evaluations(60_000),
            &mut (),
            &mut SplitMix64::new(3),
        );
        assert!(outcome.best_cost() < 4.0, "{}", outcome.best_cost());
    }

    /// Same seed, same run; exact budget.
    #[test]
    fn deterministic() {
        let run = || {
            let mut rades = Rades::new(Config::default(), Bounds::uniform(3, -1.0, 1.0));
            let mut objective = |x: &Vec<f64>| x.iter().sum::<f64>();
            rades.run(
                &mut objective,
                &mut Limits::evaluations(1_000),
                &mut (),
                &mut SplitMix64::new(5),
            )
        };

        assert_eq!(run(), run());
        assert_eq!(run().stats.evaluations, 1_000);
    }
}
