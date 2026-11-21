//! Differential evolution (Storn and Price 1997) on real vectors, with Latin hypercube
//! initialisation and a warm-start library (Yue et al. 2023), and RADES, the rank-based DE
//! with a success archive (Parque 2024), in [`rades`].
//!
//! ```
//! use evolutionary_planning::de::{Config, De};
//! use evolutionary_planning::operators::real::Bounds;
//! use evolutionary_planning::problems::single::rosenbrock;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut de = De::new(Config::default(), Bounds::uniform(2, -2.0, 2.0));
//! let mut objective = |x: &Vec<f64>| rosenbrock(x);
//! let outcome = de.run(&mut objective, &mut Limits::evaluations(20_000), &mut (), &mut SplitMix64::new(1));
//! assert!(outcome.best_cost() < 1e-8);
//! ```

pub mod rades;
mod warm_start;

use alloc::vec::Vec;

use rand_core::Rng;
pub use warm_start::WarmStartLibrary;

use crate::evaluator::Evaluator;
use crate::operators::Initializer;
use crate::operators::real::{Bounds, latin_hypercube};
use crate::rng::{below, distinct_indices, unit_f64};
use crate::{Budget, Individual, Objective, Observer, Outcome, Stats, Stop};

/// How the mutant vector is formed (the `x/y` of `DE/x/y/z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// `v = x_r1 + F (x_r2 - x_r3)`: the robust default.
    Rand1,
    /// `v = x_best + F (x_r1 - x_r2)`: faster, greedier.
    Best1,
    /// `v = x_i + F (x_best - x_i) + F (x_r1 - x_r2)`.
    CurrentToBest1,
    /// `v = x_r1 + F (x_r2 - x_r3) + F (x_r4 - x_r5)`: more exploratory.
    Rand2,
}

/// How the trial vector takes coordinates from the mutant (the `z` of `DE/x/y/z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverKind {
    /// Each coordinate from the mutant with probability `CR`, one forced coordinate always.
    Binomial,
    /// A contiguous (cyclic) run of coordinates from the mutant, whose length is geometric
    /// with continuation probability `CR`.
    Exponential,
}

/// How the initial population is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initialization {
    /// Independent uniform points in the bounds.
    Uniform,
    /// A Latin hypercube sample: one point per stratum and coordinate (Yue et al. 2023).
    LatinHypercube,
}

/// Parameters of a [`De`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The population size `NP`, at least `6` (`Rand2` needs five distinct partners).
    pub population_size: usize,
    /// The differential weight `F`, typically in `(0, 1]`.
    pub f: f64,
    /// The crossover rate `CR` in `[0, 1]`.
    pub cr: f64,
    /// The mutation strategy.
    pub strategy: Strategy,
    /// Binomial or exponential crossover.
    pub crossover: CrossoverKind,
    /// Uniform or Latin hypercube initialisation.
    pub initialization: Initialization,
}

impl Default for Config {
    /// `DE/rand/1/bin` with `NP = 40`, `F = 0.5`, `CR = 0.9` and Latin hypercube
    /// initialisation.
    fn default() -> Self {
        Self {
            population_size: 40,
            f: 0.5,
            cr: 0.9,
            strategy: Strategy::Rand1,
            crossover: CrossoverKind::Binomial,
            initialization: Initialization::LatinHypercube,
        }
    }
}

/// Differential evolution within box bounds. A trial coordinate that leaves the box is set
/// halfway between the target's coordinate and the violated bound ("bounce back"), which
/// keeps the population inside without piling it up on the boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct De {
    /// The run parameters.
    pub config: Config,
    /// The search box.
    pub bounds: Bounds,
}

impl De {
    /// Creates a DE run.
    #[must_use]
    pub fn new(config: Config, bounds: Bounds) -> Self {
        Self { config, bounds }
    }

    /// Runs from a fresh initial population until the budget stops it.
    ///
    /// # Panics
    ///
    /// Panics if the population is smaller than `6`.
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
        self.run_from(Vec::new(), objective, budget, observer, rng)
    }

    /// Runs with `seeds` (for example from a [`WarmStartLibrary`]) as the first members of
    /// the initial population (truncated to the population size, clamped into the bounds);
    /// the rest is drawn by [`Config::initialization`].
    ///
    /// # Panics
    ///
    /// Panics if the population is smaller than `6`.
    pub fn run_from<O, B, Obs, R>(
        &mut self,
        seeds: Vec<Vec<f64>>,
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
            self.config.population_size >= 6,
            "DE needs a population of at least six"
        );

        let mut evaluator = Evaluator::new(objective, budget, observer);
        let genomes = self.initial_genomes(seeds, rng);
        let Some(mut population) = evaluate_all(genomes, &mut evaluator) else {
            return evaluator.outcome(Vec::new(), Stop::Budget);
        };

        evaluator.end_generation(crate::ga::population_best(&population));
        loop {
            let best = best_index(&population);
            let mut next = population.clone();
            for i in 0..population.len() {
                let trial = self.trial(&population, i, best, rng);
                let Some(cost) = evaluator.evaluate(&trial) else {
                    return evaluator.outcome(population, Stop::Budget);
                };

                if cost <= population[i].cost {
                    next[i] = Individual {
                        genome: trial,
                        cost,
                    };
                }
            }

            population = next;
            evaluator.end_generation(crate::ga::population_best(&population));
        }
    }

    /// Seeds followed by uniform or Latin hypercube points.
    fn initial_genomes<R: Rng + ?Sized>(
        &self,
        mut seeds: Vec<Vec<f64>>,
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        let size = self.config.population_size;
        seeds.truncate(size);
        for seed in &mut seeds {
            self.bounds.clamp(seed);
        }

        let missing = size - seeds.len();
        match self.config.initialization {
            Initialization::Uniform => {
                let mut bounds = self.bounds.clone();
                seeds.extend((0..missing).map(|_| bounds.initialize(rng)));
            }
            Initialization::LatinHypercube => {
                seeds.extend(latin_hypercube(&self.bounds, missing, rng));
            }
        }

        seeds
    }

    /// The trial vector for target `i`: mutation, crossover, bounce-back.
    fn trial<R: Rng + ?Sized>(
        &self,
        population: &[Individual<Vec<f64>>],
        i: usize,
        best: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        let mutant = self.mutant(population, i, best, rng);
        let target = &population[i].genome;
        let mut trial = cross(target, &mutant, self.config.cr, self.config.crossover, rng);
        bounce_back(&mut trial, target, &self.bounds);

        trial
    }

    /// The mutant vector of the configured strategy.
    #[expect(
        clippy::many_single_char_names,
        reason = "the textbook names of the DE formulas"
    )]
    fn mutant<R: Rng + ?Sized>(
        &self,
        population: &[Individual<Vec<f64>>],
        i: usize,
        best: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        let x = |k: usize| &population[k].genome;
        let f = self.config.f;
        let n = self.bounds.dimension();
        let r = distinct_indices(rng, population.len(), 5, Some(i));
        (0..n)
            .map(|d| match self.config.strategy {
                Strategy::Rand1 => x(r[0])[d] + f * (x(r[1])[d] - x(r[2])[d]),
                Strategy::Best1 => x(best)[d] + f * (x(r[0])[d] - x(r[1])[d]),
                Strategy::CurrentToBest1 => {
                    x(i)[d] + f * (x(best)[d] - x(i)[d]) + f * (x(r[0])[d] - x(r[1])[d])
                }
                Strategy::Rand2 => {
                    x(r[0])[d] + f * (x(r[1])[d] - x(r[2])[d]) + f * (x(r[3])[d] - x(r[4])[d])
                }
            })
            .collect()
    }
}

/// Evaluates every genome; `None` if the budget ran out.
fn evaluate_all<O, B, Obs>(
    genomes: Vec<Vec<f64>>,
    evaluator: &mut Evaluator<'_, Vec<f64>, O, B, Obs>,
) -> Option<Vec<Individual<Vec<f64>>>>
where
    O: Objective<Vec<f64>> + ?Sized,
    B: Budget<Stats> + ?Sized,
    Obs: Observer + ?Sized,
{
    genomes
        .into_iter()
        .map(|g| evaluator.individual(g))
        .collect()
}

/// The index of the cheapest individual (the first among equals).
pub(crate) fn best_index<G>(population: &[Individual<G>]) -> usize {
    let mut best = 0;
    for (k, individual) in population.iter().enumerate() {
        if individual.cost < population[best].cost {
            best = k;
        }
    }

    best
}

/// Binomial or exponential crossover of `target` with `mutant`.
pub(crate) fn cross<R: Rng + ?Sized>(
    target: &[f64],
    mutant: &[f64],
    cr: f64,
    kind: CrossoverKind,
    rng: &mut R,
) -> Vec<f64> {
    let n = target.len();
    let mut trial = target.to_vec();
    let forced = below(rng, n);
    match kind {
        CrossoverKind::Binomial => {
            for d in 0..n {
                if d == forced || unit_f64(rng) < cr {
                    trial[d] = mutant[d];
                }
            }
        }
        CrossoverKind::Exponential => {
            let mut d = forced;
            for _ in 0..n {
                trial[d] = mutant[d];
                d = (d + 1) % n;
                if unit_f64(rng) >= cr {
                    break;
                }
            }
        }
    }

    trial
}

/// Moves coordinates outside the box halfway between the target and the violated bound.
pub(crate) fn bounce_back(trial: &mut [f64], target: &[f64], bounds: &Bounds) {
    for (d, value) in trial.iter_mut().enumerate() {
        let (low, high) = (bounds.lower()[d], bounds.upper()[d]);
        if *value < low {
            *value = f64::midpoint(low, target[d]);
        } else if *value > high {
            *value = f64::midpoint(high, target[d]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::single::{rastrigin, sphere};
    use crate::{Limits, SplitMix64};

    /// Every strategy and crossover solves the sphere.
    #[test]
    fn strategies_solve_sphere() {
        for strategy in [
            Strategy::Rand1,
            Strategy::Best1,
            Strategy::CurrentToBest1,
            Strategy::Rand2,
        ] {
            for crossover in [CrossoverKind::Binomial, CrossoverKind::Exponential] {
                let config = Config {
                    strategy,
                    crossover,
                    cr: if crossover == CrossoverKind::Binomial {
                        0.9
                    } else {
                        0.95
                    },
                    ..Config::default()
                };
                let mut de = De::new(config, Bounds::uniform(5, -5.0, 5.0));
                let mut objective = |x: &Vec<f64>| sphere(x);
                let outcome = de.run(
                    &mut objective,
                    &mut Limits::evaluations(40_000),
                    &mut (),
                    &mut SplitMix64::new(2),
                );
                assert!(
                    outcome.best_cost() < 1e-6,
                    "{config:?}: {}",
                    outcome.best_cost()
                );
            }
        }
    }

    /// DE with a low crossover rate solves separable Rastrigin.
    #[test]
    fn separable_rastrigin() {
        let config = Config {
            cr: 0.1,
            ..Config::default()
        };
        let mut de = De::new(config, Bounds::uniform(5, -5.12, 5.12));
        let mut objective = |x: &Vec<f64>| rastrigin(x);
        let outcome = de.run(
            &mut objective,
            &mut Limits::evaluations(100_000),
            &mut (),
            &mut SplitMix64::new(4),
        );

        assert!(outcome.best_cost() < 1e-6, "{}", outcome.best_cost());
    }

    /// The population never leaves the bounds; same seed, same run; exact budget.
    #[test]
    fn bounded_and_deterministic() {
        let bounds = Bounds::uniform(3, 0.0, 1.0);
        let run = || {
            let mut de = De::new(Config::default(), bounds.clone());
            // The optimum is outside the box, so trials keep hitting the bound.
            let mut objective = |x: &Vec<f64>| x.iter().map(|v| (v + 3.0) * (v + 3.0)).sum::<f64>();
            de.run(
                &mut objective,
                &mut Limits::evaluations(2_001),
                &mut (),
                &mut SplitMix64::new(8),
            )
        };
        let outcome = run();

        assert_eq!(outcome, run());
        assert_eq!(outcome.stats.evaluations, 2_001);
        assert!(
            outcome
                .population
                .iter()
                .all(|i| bounds.contains(&i.genome))
        );
    }
}
