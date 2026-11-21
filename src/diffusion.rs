//! GLEAM's diffusion model (Jakob 2021, section 7): a structured population on a ring, where
//! every individual mates only within its deme of neighbours, so good genes spread slowly,
//! niches form and premature convergence is counteracted.
//!
//! ```mermaid
//! graph LR
//!     pick[next individual on the ring] --> partner[partner from its deme, linear ranking]
//!     partner --> breed[offspring: crossover, mutation, repair]
//!     breed --> best[best offspring, optional local search]
//!     best --> accept{acceptance rule}
//!     accept -->|accepted| replace[replace the individual]
//!     accept -->|rejected| pick
//!     replace --> pick
//! ```
//!
//! A generation visits every individual once, in ring order, updating in place. Two
//! stagnation counters serve as the termination criteria Jakob recommends: generations in
//! which no deme's best improved, and generations without any accepted offspring.
//!
//! ```
//! use evolutionary_planning::diffusion::{Config, Diffusion};
//! use evolutionary_planning::ga::Operators;
//! use evolutionary_planning::operators::{NoRepair, discrete::{BitFlip, RandomBits}, sequence::UniformCrossover};
//! use evolutionary_planning::operators::selection::LinearRanking;
//! use evolutionary_planning::problems::single::one_max;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut diffusion = Diffusion::new(Config::default(), Operators {
//!     init: RandomBits { len: 30 },
//!     selection: LinearRanking::default(),
//!     crossover: UniformCrossover::default(),
//!     mutation: BitFlip::default(),
//!     repair: NoRepair,
//! });
//! let mut objective = |b: &Vec<bool>| one_max(b);
//! let outcome = diffusion.run(&mut objective, &mut Limits::evaluations(20_000).with_target(0.0), &mut (), &mut SplitMix64::new(1));
//! assert_eq!(outcome.best_cost(), 0.0);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::evaluator::Evaluator;
use crate::ga::{NoRefine, Operators, Refine, population_best};
use crate::operators::{Crossover, Initializer, Mutation, Repair, Selection};
use crate::rng::chance;
use crate::{Budget, Individual, Objective, Observer, Outcome, Stats, Stop};

/// When the best offspring of a mating replaces the individual at the deme's centre.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acceptance {
    /// Always.
    Always,
    /// If it is better than the worst individual of the deme ("local least").
    LocalLeast,
    /// If it is better than the individual it replaces.
    BetterParent,
    /// If it is better than the deme's worst, and, when the individual is the deme's best,
    /// also better than it ("local elitist"); the default.
    LocalElitist,
}

/// Parameters of a [`Diffusion`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The number of individuals on the ring, at least twice the deme size.
    pub population_size: usize,
    /// The deme size (the individual and its neighbours), odd, at least `3`; Jakob reports
    /// 7 to 11 as typical. An even value is rounded up.
    pub deme_size: usize,
    /// Offspring produced per mating; the best one competes for acceptance.
    pub offspring: usize,
    /// The probability that a mating uses crossover (otherwise the offspring are mutants of
    /// the individual).
    pub crossover_rate: f64,
    /// The acceptance rule.
    pub acceptance: Acceptance,
    /// Stop after this many consecutive generations in which no deme's best improved.
    pub max_without_improvement: Option<u64>,
    /// Stop after this many consecutive generations without an accepted offspring.
    pub max_without_acceptance: Option<u64>,
}

impl Default for Config {
    /// 60 individuals, demes of 9, two offspring per mating, crossover rate 0.6, local
    /// elitist acceptance, no stagnation limits.
    fn default() -> Self {
        Self {
            population_size: 60,
            deme_size: 9,
            offspring: 2,
            crossover_rate: 0.6,
            acceptance: Acceptance::LocalElitist,
            max_without_improvement: None,
            max_without_acceptance: None,
        }
    }
}

/// The diffusion-model GA; the [`Operators::selection`] picks the partner within the deme
/// (linear ranking is GLEAM's choice).
#[derive(Debug, Clone, PartialEq)]
pub struct Diffusion<I, S, X, M, P> {
    /// The run parameters.
    pub config: Config,
    /// The breeding operators.
    pub operators: Operators<I, S, X, M, P>,
}

/// What one generation changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Progress {
    /// Some accepted offspring beat its deme's best.
    improved: bool,
    /// Some offspring was accepted.
    accepted: bool,
}

impl<I, S, X, M, P> Diffusion<I, S, X, M, P> {
    /// Creates a diffusion-model run.
    #[must_use]
    pub fn new(config: Config, operators: Operators<I, S, X, M, P>) -> Self {
        Self { config, operators }
    }

    /// Runs until the budget or a stagnation limit stops it.
    ///
    /// # Panics
    ///
    /// Panics if the population is smaller than twice the deme size.
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
        S: Selection,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let mut evaluator = Evaluator::new(objective, budget, observer);

        self.evolve(&mut evaluator, &mut NoRefine, rng)
    }

    /// Runs HyGLEAM: the diffusion model with `memetic` local search on the best offspring of
    /// every mating.
    ///
    /// # Panics
    ///
    /// Panics if the population is smaller than twice the deme size.
    pub fn run_memetic<G, L, O, B, Obs, R>(
        &mut self,
        memetic: &mut crate::memetic::Memetic<L>,
        objective: &mut O,
        budget: &mut B,
        observer: &mut Obs,
        rng: &mut R,
    ) -> Outcome<G>
    where
        G: Clone,
        I: Initializer<G>,
        S: Selection,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        L: crate::memetic::LocalSearch<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let mut evaluator = Evaluator::new(objective, budget, observer);

        self.evolve(&mut evaluator, memetic, rng)
    }

    /// The generation loop with its stagnation counters.
    fn evolve<G, O, B, Obs, F, R>(
        &mut self,
        evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        refine: &mut F,
        rng: &mut R,
    ) -> Outcome<G>
    where
        G: Clone,
        I: Initializer<G>,
        S: Selection,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        F: Refine<G>,
        R: Rng + ?Sized,
    {
        let radius = self.config.deme_size.max(3) / 2;
        assert!(
            self.config.population_size >= 2 * (2 * radius + 1),
            "the ring must hold at least two demes"
        );

        let mut population = Vec::with_capacity(self.config.population_size);
        for _ in 0..self.config.population_size {
            let Some(member) = evaluator.individual(self.operators.init.initialize(rng)) else {
                return evaluator.outcome(population, Stop::Budget);
            };
            population.push(member);
        }

        evaluator.end_generation(population_best(&population));
        let (mut without_improvement, mut without_acceptance) = (0, 0);
        loop {
            let Some(progress) = self.generation(&mut population, radius, evaluator, refine, rng)
            else {
                return evaluator.outcome(population, Stop::Budget);
            };

            evaluator.end_generation(population_best(&population));
            without_improvement = if progress.improved {
                0
            } else {
                without_improvement + 1
            };
            without_acceptance = if progress.accepted {
                0
            } else {
                without_acceptance + 1
            };
            let stagnated = self
                .config
                .max_without_improvement
                .is_some_and(|m| without_improvement >= m)
                || self
                    .config
                    .max_without_acceptance
                    .is_some_and(|m| without_acceptance >= m);
            if stagnated {
                return evaluator.outcome(population, Stop::Stagnation);
            }
        }
    }

    /// One pass over the ring; `None` if the budget ran out.
    fn generation<G, O, B, Obs, F, R>(
        &mut self,
        population: &mut [Individual<G>],
        radius: usize,
        evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        refine: &mut F,
        rng: &mut R,
    ) -> Option<Progress>
    where
        G: Clone,
        S: Selection,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        F: Refine<G>,
        R: Rng + ?Sized,
    {
        let n = population.len();
        let mut progress = Progress::default();
        let mut picked = Vec::with_capacity(1);
        for centre in 0..n {
            let neighbours: Vec<usize> = (1..=radius)
                .flat_map(|d| [(centre + n - d) % n, (centre + d) % n])
                .collect();
            let costs: Vec<f64> = neighbours.iter().map(|&k| population[k].cost).collect();
            picked.clear();
            self.operators.selection.select(&costs, 1, rng, &mut picked);
            let partner = neighbours[picked[0]];
            let child = self.best_offspring(
                &population[centre].genome,
                &population[partner].genome,
                evaluator,
                rng,
            )?;
            let child = refine.refine(child, evaluator, rng);
            let deme_worst = costs
                .iter()
                .copied()
                .fold(population[centre].cost, f64::max);
            let deme_best = costs
                .iter()
                .copied()
                .fold(population[centre].cost, f64::min);
            if self.accepts(&child, &population[centre], deme_worst, deme_best) {
                progress.accepted = true;
                progress.improved |= child.cost < deme_best;
                population[centre] = child;
            }
        }

        Some(progress)
    }

    /// Breeds `offspring` children of the individual and its partner and returns the best.
    fn best_offspring<G, O, B, Obs, R>(
        &mut self,
        own: &G,
        partner: &G,
        evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        rng: &mut R,
    ) -> Option<Individual<G>>
    where
        G: Clone,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let mut best: Option<Individual<G>> = None;
        let mut produced = 0;
        while produced < self.config.offspring.max(1) {
            let children = if chance(rng, self.config.crossover_rate) {
                self.operators.crossover.crossover(own, partner, rng)
            } else {
                [own.clone(), own.clone()]
            };

            for mut child in children {
                if produced == self.config.offspring.max(1) {
                    break;
                }

                self.operators.mutation.mutate(&mut child, rng);
                self.operators.repair.repair(&mut child, rng);
                let individual = evaluator.individual(child)?;
                produced += 1;
                if best.as_ref().is_none_or(|b| individual.cost < b.cost) {
                    best = Some(individual);
                }
            }
        }

        best
    }

    /// The acceptance rule.
    fn accepts<G>(
        &self,
        child: &Individual<G>,
        parent: &Individual<G>,
        deme_worst: f64,
        deme_best: f64,
    ) -> bool {
        match self.config.acceptance {
            Acceptance::Always => true,
            Acceptance::LocalLeast => child.cost < deme_worst,
            Acceptance::BetterParent => child.cost < parent.cost,
            Acceptance::LocalElitist => {
                let parent_is_best = parent.cost <= deme_best;
                child.cost < deme_worst && (!parent_is_best || child.cost < parent.cost)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memetic::{Memetic, Rosenbrock};
    use crate::operators::NoRepair;
    use crate::operators::real::{Bounds, Polynomial, Sbx};
    use crate::operators::selection::LinearRanking;
    use crate::problems::single::{rastrigin, rosenbrock};
    use crate::{Limits, SplitMix64};

    /// A real-coded diffusion model.
    fn real(
        acceptance: Acceptance,
        bounds: &Bounds,
    ) -> Diffusion<Bounds, LinearRanking, Sbx, Polynomial, NoRepair> {
        Diffusion::new(
            Config {
                acceptance,
                ..Config::default()
            },
            Operators {
                init: bounds.clone(),
                selection: LinearRanking::default(),
                crossover: Sbx {
                    bounds: bounds.clone(),
                    eta: 15.0,
                },
                mutation: Polynomial {
                    bounds: bounds.clone(),
                    eta: 20.0,
                    rate: None,
                },
                repair: NoRepair,
            },
        )
    }

    /// Every acceptance rule makes progress on Rastrigin; the elitist ones (better parent,
    /// local elitist) never lose the best individual.
    #[test]
    fn acceptance_rules() {
        let bounds = Bounds::uniform(4, -5.12, 5.12);
        for acceptance in [
            Acceptance::Always,
            Acceptance::LocalLeast,
            Acceptance::BetterParent,
            Acceptance::LocalElitist,
        ] {
            let mut objective = |x: &Vec<f64>| rastrigin(x);
            let outcome = real(acceptance, &bounds).run(
                &mut objective,
                &mut Limits::evaluations(40_000),
                &mut (),
                &mut SplitMix64::new(2),
            );
            assert!(
                outcome.best_cost() < 3.0,
                "{acceptance:?}: {}",
                outcome.best_cost()
            );
            if matches!(
                acceptance,
                Acceptance::BetterParent | Acceptance::LocalElitist
            ) {
                assert_eq!(
                    outcome.population[0].cost,
                    outcome.best_cost(),
                    "{acceptance:?}"
                );
            }
        }
    }

    /// The stagnation criteria end a converged run before the budget.
    #[test]
    fn stagnation_stops() {
        let bounds = Bounds::uniform(2, -1.0, 1.0);
        let mut diffusion = real(Acceptance::BetterParent, &bounds);
        diffusion.config.max_without_acceptance = Some(5);
        let mut flat = |_: &Vec<f64>| 1.0;
        let outcome = diffusion.run(
            &mut flat,
            &mut Limits::evaluations(1_000_000),
            &mut (),
            &mut SplitMix64::new(1),
        );

        assert_eq!(outcome.stop, Stop::Stagnation);
        assert!(outcome.stats.evaluations < 1_000);
    }

    /// HyGLEAM (diffusion model plus adaptive Rosenbrock search) solves Rosenbrock in 5D.
    #[test]
    fn hygleam_solves_rosenbrock() {
        let bounds = Bounds::uniform(5, -2.0, 2.0);
        let mut diffusion = real(Acceptance::LocalElitist, &bounds);
        diffusion.config.population_size = 20;
        diffusion.config.deme_size = 5;
        let mut memetic = Memetic::adaptive(Rosenbrock::default());
        let mut objective = |x: &Vec<f64>| rosenbrock(x);
        let outcome = diffusion.run_memetic(
            &mut memetic,
            &mut objective,
            &mut Limits::evaluations(40_000),
            &mut (),
            &mut SplitMix64::new(3),
        );

        assert!(outcome.best_cost() < 1e-6, "{}", outcome.best_cost());
    }
}
