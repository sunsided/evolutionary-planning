//! NSGA-II (Deb, Pratap, Agarwal and Meyarivan 2002): elitist multi-objective search by
//! non-dominated sorting and crowding distance, with Deb's constrained domination.
//!
//! ```mermaid
//! graph LR
//!     P[population P] --> T[binary crowded tournament]
//!     T --> V[crossover, mutation, repair]
//!     V --> Q[offspring Q, evaluated]
//!     Q --> R["R = P ∪ Q"]
//!     R --> S[non-dominated sort]
//!     S --> F[fill by fronts, last front by crowding]
//!     F --> P
//! ```
//!
//! ```
//! use evolutionary_planning::nsga2::{Config, Nsga2};
//! use evolutionary_planning::operators::real::{Bounds, Polynomial, Sbx};
//! use evolutionary_planning::operators::{NoRepair, Variation};
//! use evolutionary_planning::pareto::hypervolume;
//! use evolutionary_planning::problems::multi::Zdt;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let bounds = Bounds::uniform(10, 0.0, 1.0);
//! let mut nsga2 = Nsga2::new(Config::default(), Variation {
//!     init: bounds.clone(),
//!     crossover: Sbx { bounds: bounds.clone(), eta: 15.0 },
//!     mutation: Polynomial { bounds, eta: 20.0, rate: None },
//!     repair: NoRepair,
//! });
//! let outcome = nsga2.run(&mut Zdt::Zdt1, &mut Limits::evaluations(20_000), &mut (), &mut SplitMix64::new(1));
//! let points: Vec<&[f64]> = outcome.front.iter().map(|e| e.objectives.as_slice()).collect();
//! assert!(hypervolume(&points, &[1.1, 1.1]) > 0.85);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::multi::{MultiEvaluator, MultiOutcome};
use crate::operators::{Crossover, Initializer, Mutation, Repair, Variation};
use crate::pareto::{Evaluated, crowded_order, crowding_distance, non_dominated_sort};
use crate::rng::below;
use crate::{Budget, MultiObjective, Observer, Stats, Stop};

/// Parameters of an [`Nsga2`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The population size `N`, at least `2`.
    pub population_size: usize,
    /// The probability that a pair of parents is recombined.
    pub crossover_rate: f64,
    /// The probability that a child is mutated at all.
    pub mutation_rate: f64,
}

impl Default for Config {
    /// `N = 100`, crossover rate `0.9`, every child mutated (the per-gene rate is the
    /// mutation operator's).
    fn default() -> Self {
        Self {
            population_size: 100,
            crossover_rate: 0.9,
            mutation_rate: 1.0,
        }
    }
}

/// NSGA-II over any genome.
#[derive(Debug, Clone, PartialEq)]
pub struct Nsga2<I, X, M, P> {
    /// The run parameters.
    pub config: Config,
    /// The variation operators.
    pub variation: Variation<I, X, M, P>,
}

impl<I, X, M, P> Nsga2<I, X, M, P> {
    /// Creates an NSGA-II run.
    #[must_use]
    pub fn new(config: Config, variation: Variation<I, X, M, P>) -> Self {
        Self { config, variation }
    }

    /// Runs until the budget stops it; returns the final population and its first front.
    ///
    /// # Panics
    ///
    /// Panics if the population size is below `2`.
    pub fn run<G, Pr, B, Obs, R>(
        &mut self,
        problem: &mut Pr,
        budget: &mut B,
        observer: &mut Obs,
        rng: &mut R,
    ) -> MultiOutcome<G>
    where
        G: Clone,
        I: Initializer<G>,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        Pr: MultiObjective<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let n = self.config.population_size;
        assert!(n >= 2, "NSGA-II needs at least two individuals");

        let mut evaluator = MultiEvaluator::new(problem, budget, observer);
        let mut population = Vec::with_capacity(n);
        for _ in 0..n {
            let Some(member) = evaluator.evaluate(self.variation.init.initialize(rng)) else {
                return evaluator.outcome(population, Stop::Budget);
            };
            population.push(member);
        }

        let (mut rank, mut crowd) = rank_and_crowd(&population);
        evaluator.end_generation();
        loop {
            let mut offspring = Vec::with_capacity(n);
            while offspring.len() < n {
                let a = tournament(&rank, &crowd, rng);
                let b = tournament(&rank, &crowd, rng);
                let children = self.variation.breed(
                    &population[a].genome,
                    &population[b].genome,
                    self.config.crossover_rate,
                    self.config.mutation_rate,
                    rng,
                );
                for child in children {
                    if offspring.len() == n {
                        break;
                    }

                    let Some(member) = evaluator.evaluate(child) else {
                        return evaluator.outcome(population, Stop::Budget);
                    };
                    offspring.push(member);
                }
            }

            population.extend(offspring);
            population = environmental_selection(population, n);
            (rank, crowd) = rank_and_crowd(&population);
            evaluator.end_generation();
        }
    }
}

/// The binary crowded tournament: the better of two uniform picks by rank, then crowding.
fn tournament<R: Rng + ?Sized>(rank: &[usize], crowd: &[f64], rng: &mut R) -> usize {
    let a = below(rng, rank.len());
    let b = below(rng, rank.len());

    if crowded_order(rank[b], crowd[b], rank[a], crowd[a]).is_lt() {
        b
    } else {
        a
    }
}

/// The front rank and crowding distance of every member.
fn rank_and_crowd<G>(population: &[Evaluated<G>]) -> (Vec<usize>, Vec<f64>) {
    let points: Vec<&[f64]> = population.iter().map(|e| e.objectives.as_slice()).collect();
    let violations: Vec<f64> = population.iter().map(|e| e.violation).collect();
    let mut rank = alloc::vec![0; population.len()];
    let mut crowd = alloc::vec![0.0; population.len()];
    for (r, front) in non_dominated_sort(&points, &violations).iter().enumerate() {
        for (slot, d) in front.iter().zip(crowding_distance(&points, front)) {
            rank[*slot] = r;
            crowd[*slot] = d;
        }
    }

    (rank, crowd)
}

/// Keeps the `n` best of `combined` by front, then crowding distance within the last front.
pub(crate) fn environmental_selection<G>(
    combined: Vec<Evaluated<G>>,
    n: usize,
) -> Vec<Evaluated<G>> {
    let points: Vec<&[f64]> = combined.iter().map(|e| e.objectives.as_slice()).collect();
    let violations: Vec<f64> = combined.iter().map(|e| e.violation).collect();
    let mut chosen = Vec::with_capacity(n);
    for front in non_dominated_sort(&points, &violations) {
        if chosen.len() + front.len() <= n {
            chosen.extend_from_slice(&front);
            if chosen.len() == n {
                break;
            }
            continue;
        }

        let crowd = crowding_distance(&points, &front);
        let mut order: Vec<usize> = (0..front.len()).collect();
        order.sort_by(|&a, &b| crowd[b].total_cmp(&crowd[a]).then(a.cmp(&b)));
        chosen.extend(
            order
                .into_iter()
                .take(n - chosen.len())
                .map(|slot| front[slot]),
        );
        break;
    }

    chosen.sort_unstable();
    let mut keep = alloc::vec![false; combined.len()];
    for &i in &chosen {
        keep[i] = true;
    }

    combined
        .into_iter()
        .zip(keep)
        .filter_map(|(e, k)| k.then_some(e))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests;
