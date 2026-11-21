//! Genetic algorithms: generational with elitism, and steady state.
//!
//! ```mermaid
//! graph LR
//!     init[initialise and evaluate] --> select[select parents]
//!     select --> cross[crossover with rate]
//!     cross --> mutate[mutate with rate]
//!     mutate --> repair[repair]
//!     repair --> eval[evaluate]
//!     eval --> refine[optional local search]
//!     refine --> replace[replace: generational with elites, or steady state]
//!     replace --> select
//! ```
//!
//! # Examples
//!
//! OneMax on 32 bits:
//!
//! ```
//! use evolutionary_planning::ga::{Config, Ga, Operators};
//! use evolutionary_planning::operators::{discrete::{BitFlip, RandomBits}, selection::Tournament, sequence::UniformCrossover, NoRepair};
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut ga = Ga::new(Config::default(), Operators {
//!     init: RandomBits { len: 32 },
//!     selection: Tournament { size: 2 },
//!     crossover: UniformCrossover::default(),
//!     mutation: BitFlip::default(),
//!     repair: NoRepair,
//! });
//! let mut zeros = |bits: &Vec<bool>| bits.iter().filter(|&&b| !b).count() as f64;
//! let outcome = ga.run(&mut zeros, &mut Limits::evaluations(20_000).with_target(0.0), &mut (), &mut SplitMix64::new(1));
//! assert_eq!(outcome.best_cost(), 0.0);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::evaluator::Evaluator;
use crate::operators::{Crossover, Initializer, Mutation, Repair, Selection};
use crate::rng::chance;
use crate::{Budget, Individual, Objective, Observer, Outcome, Stats, Stop};

/// How offspring enter the population.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// Each generation replaces the whole population except the
    /// [`Config::elitism`] best, which are copied unchanged.
    Generational,
    /// Each step breeds two offspring; each replaces the current worst individual. With
    /// `only_if_better`, only if it is strictly cheaper. One generation is counted per
    /// `population_size` offspring.
    SteadyState {
        /// Replace only when the offspring is strictly cheaper than the worst.
        only_if_better: bool,
    },
}

/// Parameters of a [`Ga`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The number of individuals, at least `2`.
    pub population_size: usize,
    /// Generational scheme: the number of best individuals copied unchanged into the next
    /// generation (clamped to `population_size - 1`, so every generation breeds). Ignored by
    /// the steady-state scheme, which is elitist by construction.
    pub elitism: usize,
    /// The probability that a pair of parents is recombined; otherwise the children are
    /// copies.
    pub crossover_rate: f64,
    /// The probability that a child is mutated at all; the mutation operator decides how
    /// much (for example a per-gene rate).
    pub mutation_rate: f64,
    /// Generational or steady state.
    pub scheme: Scheme,
}

impl Default for Config {
    /// 100 individuals, one elite, crossover rate 0.9, every child mutated, generational.
    fn default() -> Self {
        Self {
            population_size: 100,
            elitism: 1,
            crossover_rate: 0.9,
            mutation_rate: 1.0,
            scheme: Scheme::Generational,
        }
    }
}

/// The operators a [`Ga`] breeds with.
#[derive(Debug, Clone, PartialEq)]
pub struct Operators<I, S, X, M, P> {
    /// Creates the initial population.
    pub init: I,
    /// Picks parents.
    pub selection: S,
    /// Recombines two parents.
    pub crossover: X,
    /// Mutates a child.
    pub mutation: M,
    /// Repairs a child after variation; [`NoRepair`](crate::operators::NoRepair) if none.
    pub repair: P,
}

/// A genetic algorithm over any genome type.
#[derive(Debug, Clone, PartialEq)]
pub struct Ga<I, S, X, M, P> {
    /// The run parameters.
    pub config: Config,
    /// The breeding operators.
    pub operators: Operators<I, S, X, M, P>,
}

/// Something applied to every freshly evaluated offspring: nothing for a plain GA, a local
/// search for the memetic variant.
pub(crate) trait Refine<G> {
    /// Returns the (possibly improved) offspring; evaluations go through `evaluator`.
    fn refine<O, B, Obs, R>(
        &mut self,
        child: Individual<G>,
        evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        rng: &mut R,
    ) -> Individual<G>
    where
        G: Clone,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized;
}

/// The refinement that changes nothing.
pub(crate) struct NoRefine;

impl<G> Refine<G> for NoRefine {
    fn refine<O, B, Obs, R>(
        &mut self,
        child: Individual<G>,
        _evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        _rng: &mut R,
    ) -> Individual<G>
    where
        G: Clone,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        child
    }
}

impl<I, S, X, M, P> Ga<I, S, X, M, P> {
    /// Creates a GA.
    #[must_use]
    pub fn new(config: Config, operators: Operators<I, S, X, M, P>) -> Self {
        Self { config, operators }
    }

    /// Runs the GA from a random initial population until the budget stops it.
    ///
    /// The returned [`Outcome::best`] is the best genome ever evaluated; the population is the
    /// last complete one.
    ///
    /// # Panics
    ///
    /// Panics if [`Config::population_size`] is below `2`.
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
        self.run_from(Vec::new(), objective, budget, observer, rng)
    }

    /// Like [`Ga::run`], but the initial population starts with `seeds` (truncated to the
    /// population size) and is filled up by the initialiser.
    ///
    /// # Panics
    ///
    /// Panics if [`Config::population_size`] is below `2`.
    pub fn run_from<G, O, B, Obs, R>(
        &mut self,
        seeds: Vec<G>,
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

        self.evolve(seeds, &mut evaluator, &mut NoRefine, rng)
    }

    /// Runs the GA as a memetic algorithm: every evaluated offspring (and initial member)
    /// goes through `memetic`'s local search according to its schedule. Local search
    /// evaluations count against the same budget.
    ///
    /// # Panics
    ///
    /// Panics if [`Config::population_size`] is below `2`.
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

        self.evolve(Vec::new(), &mut evaluator, memetic, rng)
    }

    /// The shared driver of the plain and the memetic GA.
    pub(crate) fn evolve<G, O, B, Obs, F, R>(
        &mut self,
        seeds: Vec<G>,
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
        assert!(
            self.config.population_size >= 2,
            "a GA needs at least two individuals"
        );

        let Some(mut population) = self.initial_population(seeds, evaluator, refine, rng) else {
            return finish(evaluator, Vec::new());
        };

        evaluator.end_generation(population_best(&population));
        match self.config.scheme {
            Scheme::Generational => self.generational(&mut population, evaluator, refine, rng),
            Scheme::SteadyState { only_if_better } => {
                self.steady_state(&mut population, only_if_better, evaluator, refine, rng);
            }
        }

        finish(evaluator, population)
    }

    /// Creates and evaluates the initial population; `None` if the budget ran out first.
    fn initial_population<G, O, B, Obs, F, R>(
        &mut self,
        seeds: Vec<G>,
        evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        refine: &mut F,
        rng: &mut R,
    ) -> Option<Vec<Individual<G>>>
    where
        G: Clone,
        I: Initializer<G>,
        O: Objective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        F: Refine<G>,
        R: Rng + ?Sized,
    {
        let size = self.config.population_size;
        let mut genomes = seeds;
        genomes.truncate(size);
        while genomes.len() < size {
            genomes.push(self.operators.init.initialize(rng));
        }

        let mut population = Vec::with_capacity(size);
        for genome in genomes {
            let individual = evaluator.individual(genome)?;
            population.push(refine.refine(individual, evaluator, rng));
        }

        Some(population)
    }

    /// Runs generations until the budget is exhausted.
    fn generational<G, O, B, Obs, F, R>(
        &mut self,
        population: &mut Vec<Individual<G>>,
        evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        refine: &mut F,
        rng: &mut R,
    ) where
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
        let size = self.config.population_size;
        let mut parents = Vec::with_capacity(2);
        loop {
            crate::cost::sort_population(population);
            let costs: Vec<f64> = population.iter().map(|i| i.cost).collect();
            let mut next: Vec<Individual<G>> =
                population[..self.config.elitism.min(size - 1)].to_vec();
            while next.len() < size {
                parents.clear();
                self.operators
                    .selection
                    .select(&costs, 2, rng, &mut parents);
                for child in self.breed(
                    &population[parents[0]].genome,
                    &population[parents[1]].genome,
                    rng,
                ) {
                    if next.len() == size {
                        break;
                    }

                    let Some(individual) = evaluator.individual(child) else {
                        return;
                    };
                    next.push(refine.refine(individual, evaluator, rng));
                }
            }

            *population = next;
            evaluator.end_generation(population_best(population));
        }
    }

    /// Runs steady-state steps until the budget is exhausted.
    fn steady_state<G, O, B, Obs, F, R>(
        &mut self,
        population: &mut [Individual<G>],
        only_if_better: bool,
        evaluator: &mut Evaluator<'_, G, O, B, Obs>,
        refine: &mut F,
        rng: &mut R,
    ) where
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
        let mut parents = Vec::with_capacity(2);
        let mut offspring_in_generation = 0;
        loop {
            let costs: Vec<f64> = population.iter().map(|i| i.cost).collect();
            parents.clear();
            self.operators
                .selection
                .select(&costs, 2, rng, &mut parents);
            let children = self.breed(
                &population[parents[0]].genome,
                &population[parents[1]].genome,
                rng,
            );
            for child in children {
                let Some(individual) = evaluator.individual(child) else {
                    return;
                };

                let individual = refine.refine(individual, evaluator, rng);
                replace_worst(population, individual, only_if_better);
                offspring_in_generation += 1;
                if offspring_in_generation == population.len() {
                    offspring_in_generation = 0;
                    evaluator.end_generation(population_best(population));
                }
            }
        }
    }

    /// Produces two children of `a` and `b`: crossover, mutation and repair.
    fn breed<G, R>(&mut self, a: &G, b: &G, rng: &mut R) -> [G; 2]
    where
        G: Clone,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        R: Rng + ?Sized,
    {
        let mut children = if chance(rng, self.config.crossover_rate) {
            self.operators.crossover.crossover(a, b, rng)
        } else {
            [a.clone(), b.clone()]
        };

        for child in &mut children {
            if chance(rng, self.config.mutation_rate) {
                self.operators.mutation.mutate(child, rng);
            }

            self.operators.repair.repair(child, rng);
        }

        children
    }
}

/// Replaces the worst individual (the last among equals) by `newcomer`, if allowed.
fn replace_worst<G>(
    population: &mut [Individual<G>],
    newcomer: Individual<G>,
    only_if_better: bool,
) {
    let worst = population
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.cost.total_cmp(&b.1.cost).then(a.0.cmp(&b.0)))
        .map_or(0, |(index, _)| index);

    if !only_if_better || newcomer.cost < population[worst].cost {
        population[worst] = newcomer;
    }
}

/// The cheapest cost in a population.
pub(crate) fn population_best<G>(population: &[Individual<G>]) -> f64 {
    population
        .iter()
        .map(|i| i.cost)
        .fold(f64::INFINITY, f64::min)
}

/// Ends a run stopped by the budget.
fn finish<G: Clone, O, B, Obs>(
    evaluator: &mut Evaluator<'_, G, O, B, Obs>,
    population: Vec<Individual<G>>,
) -> Outcome<G>
where
    O: Objective<G> + ?Sized,
    B: Budget<Stats> + ?Sized,
    Obs: Observer + ?Sized,
{
    evaluator.outcome(population, Stop::Budget)
}

#[cfg(test)]
mod tests;
