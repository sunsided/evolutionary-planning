//! MOEA/D (Zhang and Li 2007): multi-objective search by decomposition into `N` scalar
//! subproblems, one per weight vector, each improved by mating within its neighbourhood of
//! the `T` closest weight vectors; with the neighbourhood mating probability `δ` and the
//! replacement limit `n_r` of Li and Zhang (2009).
//!
//! Constraints are handled by comparing violation first, then the scalarised cost.
//!
//! ```
//! use evolutionary_planning::moead::{Config, Decomposition, Moead};
//! use evolutionary_planning::operators::real::{Bounds, Polynomial, Sbx};
//! use evolutionary_planning::operators::{NoRepair, Variation};
//! use evolutionary_planning::pareto::{das_dennis, hypervolume};
//! use evolutionary_planning::problems::multi::Zdt;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let bounds = Bounds::uniform(10, 0.0, 1.0);
//! let mut moead = Moead::new(Config::default(), das_dennis(2, 99), Variation {
//!     init: bounds.clone(),
//!     crossover: Sbx { bounds: bounds.clone(), eta: 20.0 },
//!     mutation: Polynomial { bounds, eta: 20.0, rate: None },
//!     repair: NoRepair,
//! });
//! let outcome = moead.run(&mut Zdt::Zdt1, &mut Limits::evaluations(20_000), &mut (), &mut SplitMix64::new(1));
//! let points: Vec<&[f64]> = outcome.front.iter().map(|e| e.objectives.as_slice()).collect();
//! assert!(hypervolume(&points, &[1.1, 1.1]) > 0.85);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::multi::{MultiEvaluator, MultiOutcome, first_front};
use crate::operators::{Crossover, Initializer, Mutation, Repair, Variation};
use crate::pareto::{Evaluated, ParetoArchive};
use crate::rng::{below, chance, shuffle};
use crate::{Budget, MultiObjective, Observer, Stats, Stop};

/// How a weight vector turns an objective vector into a scalar cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decomposition {
    /// `Σ λ_i f_i`: finds only convex parts of the front.
    WeightedSum,
    /// `max λ_i |f_i - z_i|` with the ideal point `z`; zero weights are raised to `1e-6`.
    Tchebycheff,
    /// Penalty-based boundary intersection: `d1 + θ d2`, where `d1` is the distance along
    /// `λ` from `z` and `d2` the distance from that line.
    Pbi {
        /// The penalty `θ`; `5` is customary.
        theta: f64,
    },
}

impl Decomposition {
    /// The scalarised cost of `f` for weight `lambda` and ideal point `ideal`.
    #[must_use]
    pub fn cost(&self, f: &[f64], lambda: &[f64], ideal: &[f64]) -> f64 {
        match *self {
            Self::WeightedSum => f.iter().zip(lambda).map(|(a, w)| a * w).sum(),
            Self::Tchebycheff => f
                .iter()
                .zip(lambda)
                .zip(ideal)
                .map(|((a, w), z)| w.max(1e-6) * (a - z).abs())
                .fold(f64::NEG_INFINITY, f64::max),
            Self::Pbi { theta } => {
                let norm = libm::sqrt(lambda.iter().map(|w| w * w).sum());
                let d1 = f
                    .iter()
                    .zip(ideal)
                    .zip(lambda)
                    .map(|((a, z), w)| (a - z) * w)
                    .sum::<f64>()
                    / norm;
                let d2 = libm::sqrt(
                    f.iter()
                        .zip(ideal)
                        .zip(lambda)
                        .map(|((a, z), w)| {
                            let off = a - (z + d1 * w / norm);
                            off * off
                        })
                        .sum(),
                );
                d1 + theta * d2
            }
        }
    }
}

/// Parameters of a [`Moead`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The neighbourhood size `T`, at least `2` (clamped to the number of weights).
    pub neighbourhood: usize,
    /// The probability `δ` of mating within the neighbourhood (otherwise the whole population).
    pub delta: f64,
    /// The largest number of solutions one child may replace, `n_r`.
    pub max_replacements: usize,
    /// The scalarising function.
    pub decomposition: Decomposition,
    /// The probability that a pair of parents is recombined.
    pub crossover_rate: f64,
    /// The probability that a child is mutated at all.
    pub mutation_rate: f64,
    /// Keep an external Pareto archive of all evaluated non-dominated genomes, bounded to this
    /// size by crowding; `None` keeps none and reports the population's front.
    pub archive: Option<usize>,
}

impl Default for Config {
    /// `T = 20`, `δ = 0.9`, `n_r = 2`, Tchebycheff, crossover rate `1`, no archive.
    fn default() -> Self {
        Self {
            neighbourhood: 20,
            delta: 0.9,
            max_replacements: 2,
            decomposition: Decomposition::Tchebycheff,
            crossover_rate: 1.0,
            mutation_rate: 1.0,
            archive: None,
        }
    }
}

/// MOEA/D with caller-supplied weight vectors (see [`das_dennis`](crate::pareto::das_dennis)).
#[derive(Debug, Clone, PartialEq)]
pub struct Moead<I, X, M, P> {
    /// The run parameters.
    pub config: Config,
    /// One weight vector per subproblem; the population size is their number.
    pub weights: Vec<Vec<f64>>,
    /// The variation operators.
    pub variation: Variation<I, X, M, P>,
}

impl<I, X, M, P> Moead<I, X, M, P> {
    /// Creates a MOEA/D run.
    #[must_use]
    pub fn new(config: Config, weights: Vec<Vec<f64>>, variation: Variation<I, X, M, P>) -> Self {
        Self {
            config,
            weights,
            variation,
        }
    }

    /// Runs until the budget stops it; one generation is one pass over all subproblems.
    ///
    /// # Panics
    ///
    /// Panics if there are fewer than two weight vectors, or their dimension differs from the
    /// problem's objective count.
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
        let n = self.weights.len();
        assert!(n >= 2, "MOEA/D needs at least two weight vectors");
        assert!(
            self.weights.iter().all(|w| w.len() == problem.objectives()),
            "weights must have one coordinate per objective"
        );

        let neighbours = neighbourhoods(&self.weights, self.config.neighbourhood.clamp(2, n));
        let mut archive = self.config.archive.map(|cap| ParetoArchive::new(Some(cap)));
        let mut evaluator = MultiEvaluator::new(problem, budget, observer);
        let mut population = Vec::with_capacity(n);
        for _ in 0..n {
            let Some(member) = evaluator.evaluate(self.variation.init.initialize(rng)) else {
                return finish(&evaluator, population, archive);
            };
            offer(&mut archive, &member);
            population.push(member);
        }

        let mut ideal = ideal_point(&population);
        evaluator.end_generation();
        loop {
            for neighbourhood in &neighbours {
                let pool: Vec<usize> = if chance(rng, self.config.delta) {
                    neighbourhood.clone()
                } else {
                    (0..n).collect()
                };
                let child = self.child(&population, &pool, rng);
                let Some(child) = evaluator.evaluate(child) else {
                    return finish(&evaluator, population, archive);
                };

                for (z, f) in ideal.iter_mut().zip(&child.objectives) {
                    *z = z.min(*f);
                }
                offer(&mut archive, &child);
                self.replace(&mut population, &child, pool, &ideal, rng);
            }

            evaluator.end_generation();
        }
    }

    /// Breeds one child from two parents of `pool`.
    fn child<G, R>(&mut self, population: &[Evaluated<G>], pool: &[usize], rng: &mut R) -> G
    where
        G: Clone,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        R: Rng + ?Sized,
    {
        let a = pool[below(rng, pool.len())];
        let b = pool[below(rng, pool.len())];
        let [child, _] = self.variation.breed(
            &population[a].genome,
            &population[b].genome,
            self.config.crossover_rate,
            self.config.mutation_rate,
            rng,
        );

        child
    }

    /// Replaces up to `n_r` members of `pool` (in random order) that the child beats on their
    /// own subproblem.
    fn replace<G: Clone, R: Rng + ?Sized>(
        &self,
        population: &mut [Evaluated<G>],
        child: &Evaluated<G>,
        mut pool: Vec<usize>,
        ideal: &[f64],
        rng: &mut R,
    ) {
        shuffle(rng, &mut pool);
        let mut replaced = 0;
        for j in pool {
            if replaced >= self.config.max_replacements {
                break;
            }

            let current = &population[j];
            let better = if child.violation.total_cmp(&current.violation).is_ne() {
                child.violation < current.violation
            } else {
                let d = &self.config.decomposition;
                d.cost(&child.objectives, &self.weights[j], ideal)
                    <= d.cost(&current.objectives, &self.weights[j], ideal)
            };
            if better {
                population[j] = child.clone();
                replaced += 1;
            }
        }
    }
}

/// For each weight vector, the indices of the `t` closest ones (itself included), nearest
/// first, ties by index.
fn neighbourhoods(weights: &[Vec<f64>], t: usize) -> Vec<Vec<usize>> {
    weights
        .iter()
        .map(|w| {
            let mut order: Vec<(f64, usize)> = weights
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    (
                        w.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum::<f64>(),
                        j,
                    )
                })
                .collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            order.into_iter().take(t).map(|(_, j)| j).collect()
        })
        .collect()
}

/// The component-wise minimum of the population's objectives.
fn ideal_point<G>(population: &[Evaluated<G>]) -> Vec<f64> {
    let m = population[0].objectives.len();

    (0..m)
        .map(|j| {
            population
                .iter()
                .map(|e| e.objectives[j])
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// Offers a member to the external archive, if one is kept.
fn offer<G: Clone>(archive: &mut Option<ParetoArchive<G>>, member: &Evaluated<G>) {
    if let Some(archive) = archive {
        archive.insert(
            member.genome.clone(),
            member.objectives.clone(),
            member.violation,
        );
    }
}

/// Ends the run; the front comes from the archive when one is kept.
fn finish<G: Clone, Pr: ?Sized, B, Obs>(
    evaluator: &MultiEvaluator<'_, Pr, B, Obs>,
    population: Vec<Evaluated<G>>,
    archive: Option<ParetoArchive<G>>,
) -> MultiOutcome<G>
where
    B: Budget<Stats> + ?Sized,
    Obs: Observer + ?Sized,
{
    let mut outcome = evaluator.outcome(population, Stop::Budget);
    if let Some(archive) = archive {
        outcome.front = first_front(archive.members());
    }

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nsga2::tests::{brute_front, distinct};
    use crate::operators::NoRepair;
    use crate::operators::discrete::{BitFlip, RandomBits};
    use crate::operators::sequence::UniformCrossover;
    use crate::pareto::das_dennis;
    use crate::problems::multi::Knapsack;
    use crate::{Limits, SplitMix64};

    /// Every decomposition returns only brute-force optimal knapsack points; with an archive
    /// the whole front is found.
    #[test]
    fn knapsack_against_brute_force() {
        let mut problem = Knapsack::generate(12, 7);
        let expected = brute_front(&mut problem, 12);
        for decomposition in [
            Decomposition::Tchebycheff,
            Decomposition::Pbi { theta: 5.0 },
            Decomposition::WeightedSum,
        ] {
            let config = Config {
                decomposition,
                archive: Some(200),
                ..Config::default()
            };
            let mut moead = Moead::new(
                config,
                das_dennis(2, 59),
                Variation {
                    init: RandomBits { len: 12 },
                    crossover: UniformCrossover::default(),
                    mutation: BitFlip::default(),
                    repair: NoRepair,
                },
            );
            let outcome = moead.run(
                &mut problem,
                &mut Limits::evaluations(30_000),
                &mut (),
                &mut SplitMix64::new(4),
            );
            let found = distinct(&outcome.front);

            assert!(
                found.iter().all(|p| expected.contains(p)),
                "{decomposition:?}: {found:?}"
            );
            if decomposition == Decomposition::Tchebycheff {
                assert_eq!(found, expected);
            }
        }
    }

    /// Hand-checked scalarisations.
    #[test]
    fn scalarisations() {
        let f = [2.0, 1.0];
        let w = [0.5, 0.5];
        let z = [0.0, 0.0];

        assert_eq!(Decomposition::WeightedSum.cost(&f, &w, &z), 1.5);
        assert_eq!(Decomposition::Tchebycheff.cost(&f, &w, &z), 1.0);
        let pbi = Decomposition::Pbi { theta: 5.0 }.cost(&f, &w, &z);
        let d1 = 1.5 / libm::sqrt(0.5);
        let d2 = libm::sqrt(0.5);
        assert!((pbi - (d1 + 5.0 * d2)).abs() < 1e-12);
    }
}
