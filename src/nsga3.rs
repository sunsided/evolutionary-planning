//! NSGA-III (Deb and Jain 2014): NSGA-II's non-dominated sorting with reference-point based
//! niching instead of crowding, for three or more objectives.
//!
//! Environmental selection: the fronts of parents plus offspring are added until at least
//! `N` members are chosen; if the last front does not fit, the chosen set is normalised
//! (ideal point, extreme points by the achievement scalarising function, hyperplane
//! intercepts, falling back to the nadir of the first front on degenerate hyperplanes), every
//! member is associated with the reference line nearest by perpendicular distance, and the
//! remaining slots go to members of the last front in the least crowded niches. Parents are
//! chosen uniformly, as in the original.
//!
//! ```
//! use evolutionary_planning::nsga3::{Config, Nsga3};
//! use evolutionary_planning::operators::real::{Bounds, Polynomial, Sbx};
//! use evolutionary_planning::operators::{NoRepair, Variation};
//! use evolutionary_planning::pareto::das_dennis;
//! use evolutionary_planning::problems::multi::{Dtlz, DtlzKind};
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let bounds = Bounds::uniform(7, 0.0, 1.0);
//! let references = das_dennis(3, 6);
//! let mut nsga3 = Nsga3::new(Config::for_references(&references), references, Variation {
//!     init: bounds.clone(),
//!     crossover: Sbx { bounds: bounds.clone(), eta: 30.0 },
//!     mutation: Polynomial { bounds, eta: 20.0, rate: None },
//!     repair: NoRepair,
//! });
//! let mut problem = Dtlz { kind: DtlzKind::Dtlz2, objectives: 3 };
//! let outcome = nsga3.run(&mut problem, &mut Limits::evaluations(20_000), &mut (), &mut SplitMix64::new(1));
//! for member in &outcome.front {
//!     let radius: f64 = member.objectives.iter().map(|f| f * f).sum();
//!     assert!((radius.sqrt() - 1.0).abs() < 0.05);
//! }
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::multi::{MultiEvaluator, MultiOutcome};
use crate::operators::{Crossover, Initializer, Mutation, Repair, Variation};
use crate::pareto::{Evaluated, non_dominated_sort};
use crate::rng::below;
use crate::{Budget, MultiObjective, Observer, Stats, Stop};

/// Parameters of an [`Nsga3`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The population size `N`, at least `2`; about the number of reference points.
    pub population_size: usize,
    /// The probability that a pair of parents is recombined.
    pub crossover_rate: f64,
    /// The probability that a child is mutated at all.
    pub mutation_rate: f64,
}

impl Config {
    /// The paper's sizing: the smallest multiple of four not below the number of reference
    /// points; crossover rate `1`, every child mutated.
    #[must_use]
    pub fn for_references(references: &[Vec<f64>]) -> Self {
        Self {
            population_size: references.len().next_multiple_of(4).max(4),
            crossover_rate: 1.0,
            mutation_rate: 1.0,
        }
    }
}

/// NSGA-III over any genome, with caller-supplied reference points (see
/// [`das_dennis`](crate::pareto::das_dennis) and [`two_layer`](crate::pareto::two_layer)).
#[derive(Debug, Clone, PartialEq)]
pub struct Nsga3<I, X, M, P> {
    /// The run parameters.
    pub config: Config,
    /// The reference points on the normalised hyperplane, each with `m` coordinates.
    pub references: Vec<Vec<f64>>,
    /// The variation operators.
    pub variation: Variation<I, X, M, P>,
}

impl<I, X, M, P> Nsga3<I, X, M, P> {
    /// Creates an NSGA-III run.
    #[must_use]
    pub fn new(
        config: Config,
        references: Vec<Vec<f64>>,
        variation: Variation<I, X, M, P>,
    ) -> Self {
        Self {
            config,
            references,
            variation,
        }
    }

    /// Runs until the budget stops it.
    ///
    /// # Panics
    ///
    /// Panics if the population size is below `2`, there are no reference points, or their
    /// dimension differs from the problem's objective count.
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
        assert!(n >= 2, "NSGA-III needs at least two individuals");
        assert!(
            !self.references.is_empty(),
            "NSGA-III needs reference points"
        );
        assert!(
            self.references
                .iter()
                .all(|r| r.len() == problem.objectives()),
            "reference points must have one coordinate per objective"
        );

        let mut evaluator = MultiEvaluator::new(problem, budget, observer);
        let mut population = Vec::with_capacity(n);
        for _ in 0..n {
            let Some(member) = evaluator.evaluate(self.variation.init.initialize(rng)) else {
                return evaluator.outcome(population, Stop::Budget);
            };
            population.push(member);
        }

        evaluator.end_generation();
        loop {
            let mut offspring = Vec::with_capacity(n);
            while offspring.len() < n {
                let a = below(rng, population.len());
                let b = below(rng, population.len());
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
            population = self.select(population, n, rng);
            evaluator.end_generation();
        }
    }

    /// NSGA-III environmental selection of `n` members from `combined`.
    fn select<G, R: Rng + ?Sized>(
        &self,
        combined: Vec<Evaluated<G>>,
        n: usize,
        rng: &mut R,
    ) -> Vec<Evaluated<G>> {
        let points: Vec<&[f64]> = combined.iter().map(|e| e.objectives.as_slice()).collect();
        let violations: Vec<f64> = combined.iter().map(|e| e.violation).collect();
        let fronts = non_dominated_sort(&points, &violations);
        let mut chosen: Vec<usize> = Vec::with_capacity(n);
        let mut last: Vec<usize> = Vec::new();
        for front in &fronts {
            if chosen.len() + front.len() <= n {
                chosen.extend_from_slice(front);
            } else {
                last.clone_from(front);
            }

            if chosen.len() >= n || !last.is_empty() {
                break;
            }
        }

        if !last.is_empty() {
            let candidates: Vec<usize> = chosen.iter().chain(&last).copied().collect();
            let first_front = &fronts[0];
            let normalised = normalise(&points, &candidates, first_front);
            let niches = associate(&normalised, &self.references);
            let picked = niching(
                n - chosen.len(),
                chosen.len(),
                &niches,
                self.references.len(),
                rng,
            );
            chosen.extend(picked.into_iter().map(|slot| last[slot]));
        }

        keep(combined, &chosen)
    }
}

/// Keeps the members of `combined` at the indices `chosen`.
fn keep<G>(combined: Vec<Evaluated<G>>, chosen: &[usize]) -> Vec<Evaluated<G>> {
    let mut flags = alloc::vec![false; combined.len()];
    for &i in chosen {
        flags[i] = true;
    }

    combined
        .into_iter()
        .zip(flags)
        .filter_map(|(e, k)| k.then_some(e))
        .collect()
}

/// Normalises the objective vectors of `candidates` (indices into `points`): translated by
/// the ideal point and divided by the hyperplane intercepts.
fn normalise(points: &[&[f64]], candidates: &[usize], first_front: &[usize]) -> Vec<Vec<f64>> {
    let m = points[candidates[0]].len();
    let ideal: Vec<f64> = (0..m)
        .map(|j| {
            candidates
                .iter()
                .map(|&i| points[i][j])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let translated: Vec<Vec<f64>> = candidates
        .iter()
        .map(|&i| points[i].iter().zip(&ideal).map(|(f, z)| f - z).collect())
        .collect();
    let intercepts = intercepts(&translated, candidates, first_front, m);

    translated
        .into_iter()
        .map(|f| f.iter().zip(&intercepts).map(|(v, a)| v / a).collect())
        .collect()
}

/// The hyperplane intercepts through the extreme points, or the nadir of the first front
/// (per objective maximum, at least `1e-10`) when the hyperplane is degenerate.
fn intercepts(
    translated: &[Vec<f64>],
    candidates: &[usize],
    first_front: &[usize],
    m: usize,
) -> Vec<f64> {
    let extremes: Vec<Vec<f64>> = (0..m)
        .map(|axis| {
            let asf = |f: &[f64]| {
                f.iter()
                    .enumerate()
                    .map(|(j, v)| v / if j == axis { 1.0 } else { 1e-6 })
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            translated
                .iter()
                .min_by(|a, b| asf(a).total_cmp(&asf(b)))
                .cloned()
                .unwrap_or_default()
        })
        .collect();

    let nadir = || {
        (0..m)
            .map(|j| {
                translated
                    .iter()
                    .zip(candidates)
                    .filter(|(_, i)| first_front.contains(i))
                    .map(|(f, _)| f[j])
                    .fold(0.0, f64::max)
                    .max(1e-10)
            })
            .collect()
    };

    match solve(extremes, alloc::vec![1.0; m]) {
        Some(a) if a.iter().all(|&v| v > 1e-10 && v.is_finite()) => {
            let intercepts: Vec<f64> = a.iter().map(|v| 1.0 / v).collect();
            if intercepts.iter().all(|&v| v > 1e-10 && v.is_finite()) {
                intercepts
            } else {
                nadir()
            }
        }
        _ => nadir(),
    }
}

/// Solves `A x = b` by Gaussian elimination with partial pivoting; `None` if singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }

        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = alloc::vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }

    Some(x)
}

/// For each normalised point: its nearest reference line and the perpendicular distance.
fn associate(normalised: &[Vec<f64>], references: &[Vec<f64>]) -> Vec<(usize, f64)> {
    normalised
        .iter()
        .map(|f| {
            references
                .iter()
                .enumerate()
                .map(|(r, w)| (r, perpendicular_distance(f, w)))
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .unwrap_or((0, f64::INFINITY))
        })
        .collect()
}

/// The distance from `f` to the line through the origin along `w`.
fn perpendicular_distance(f: &[f64], w: &[f64]) -> f64 {
    let norm2: f64 = w.iter().map(|v| v * v).sum();
    let projection: f64 = f.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / norm2;

    libm::sqrt(
        f.iter()
            .zip(w)
            .map(|(a, b)| (a - projection * b) * (a - projection * b))
            .sum(),
    )
}

/// Picks `k` members of the last front by niche count. `niches` holds the association of the
/// already chosen members (`chosen` of them, first) followed by the last front's.
fn niching<R: Rng + ?Sized>(
    k: usize,
    chosen: usize,
    niches: &[(usize, f64)],
    references: usize,
    rng: &mut R,
) -> Vec<usize> {
    let mut counts = alloc::vec![0usize; references];
    for &(r, _) in &niches[..chosen] {
        counts[r] += 1;
    }

    let last = &niches[chosen..];
    let mut taken = alloc::vec![false; last.len()];
    let mut excluded = alloc::vec![false; references];
    let mut picked = Vec::with_capacity(k);
    while picked.len() < k {
        let Some(min_count) = (0..references)
            .filter(|&r| !excluded[r])
            .map(|r| counts[r])
            .min()
        else {
            break;
        };

        let candidates: Vec<usize> = (0..references)
            .filter(|&r| !excluded[r] && counts[r] == min_count)
            .collect();
        let r = candidates[below(rng, candidates.len())];
        let members: Vec<usize> = (0..last.len())
            .filter(|&s| !taken[s] && last[s].0 == r)
            .collect();
        if members.is_empty() {
            excluded[r] = true;
            continue;
        }

        let slot = if counts[r] == 0 {
            *members
                .iter()
                .min_by(|&&a, &&b| last[a].1.total_cmp(&last[b].1).then(a.cmp(&b)))
                .unwrap_or(&members[0])
        } else {
            members[below(rng, members.len())]
        };
        taken[slot] = true;
        counts[r] += 1;
        picked.push(slot);
    }

    picked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nsga2::tests::{brute_front, distinct};
    use crate::operators::NoRepair;
    use crate::operators::discrete::{BitFlip, RandomBits};
    use crate::operators::real::{Bounds, Polynomial, Sbx};
    use crate::operators::sequence::UniformCrossover;
    use crate::pareto::das_dennis;
    use crate::problems::multi::{Dtlz, DtlzKind, Knapsack};
    use crate::{Limits, SplitMix64};

    /// On DTLZ2 with three objectives, the front lies on the unit sphere and covers most
    /// reference directions.
    #[test]
    fn dtlz2_converges_and_spreads() {
        let bounds = Bounds::uniform(12, 0.0, 1.0);
        let references = das_dennis(3, 12);
        let mut nsga3 = Nsga3::new(
            Config::for_references(&references),
            references.clone(),
            Variation {
                init: bounds.clone(),
                crossover: Sbx {
                    bounds: bounds.clone(),
                    eta: 30.0,
                },
                mutation: Polynomial {
                    bounds,
                    eta: 20.0,
                    rate: None,
                },
                repair: NoRepair,
            },
        );
        let mut problem = Dtlz {
            kind: DtlzKind::Dtlz2,
            objectives: 3,
        };
        let outcome = nsga3.run(
            &mut problem,
            &mut Limits::evaluations(60_000),
            &mut (),
            &mut SplitMix64::new(2),
        );

        let mut covered = alloc::vec![false; references.len()];
        for member in &outcome.front {
            let radius = libm::sqrt(member.objectives.iter().map(|f| f * f).sum());
            assert!((radius - 1.0).abs() < 0.01, "radius {radius}");
            let (r, _) = associate(core::slice::from_ref(&member.objectives), &references)[0];
            covered[r] = true;
        }

        let coverage = covered.iter().filter(|&&c| c).count();
        assert!(
            coverage * 10 >= references.len() * 9,
            "{coverage} of {}",
            references.len()
        );
    }

    /// On the constrained knapsack, NSGA-III returns only brute-force Pareto-optimal points.
    #[test]
    fn knapsack_points_are_optimal() {
        let mut problem = Knapsack::generate(12, 7);
        let expected = brute_front(&mut problem, 12);
        let references = das_dennis(2, 39);
        let mut nsga3 = Nsga3::new(
            Config::for_references(&references),
            references,
            Variation {
                init: RandomBits { len: 12 },
                crossover: UniformCrossover::default(),
                mutation: BitFlip::default(),
                repair: NoRepair,
            },
        );
        let outcome = nsga3.run(
            &mut problem,
            &mut Limits::evaluations(30_000),
            &mut (),
            &mut SplitMix64::new(1),
        );
        let found = distinct(&outcome.front);

        assert!(found.iter().all(|p| expected.contains(p)), "{found:?}");
        assert!(
            found.len() * 10 >= expected.len() * 8,
            "{} of {}",
            found.len(),
            expected.len()
        );
    }

    /// The linear solver and the perpendicular distance on hand-checked inputs.
    #[test]
    fn geometry_helpers() {
        let x = solve(
            alloc::vec![alloc::vec![2.0, 1.0], alloc::vec![1.0, 3.0]],
            alloc::vec![3.0, 5.0],
        )
        .expect("regular");
        assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
        assert!(
            solve(
                alloc::vec![alloc::vec![1.0, 2.0], alloc::vec![2.0, 4.0]],
                alloc::vec![1.0, 1.0]
            )
            .is_none()
        );
        assert!((perpendicular_distance(&[1.0, 1.0], &[1.0, 0.0]) - 1.0).abs() < 1e-12);
    }
}
