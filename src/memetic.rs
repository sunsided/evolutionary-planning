//! Memetic search: local searchers applied to offspring, with HyGLEAM's cost-benefit
//! adaptation of how often and how long they run (Jakob 2021, section 5; Jakob, Blume and
//! Bretthauer 2004).
//!
//! A [`Memetic`] hybrid plugs into [`Ga::run_memetic`](crate::ga::Ga::run_memetic) and
//! [`Diffusion::run_memetic`](crate::diffusion::Diffusion::run_memetic). It answers the five
//! design questions Jakob lists:
//!
//! 1. which local searcher: any [`LocalSearch`] ([`HillClimb`] for any genome,
//!    [`Rosenbrock`] for real vectors);
//! 2. whether to write the result back: Lamarckian or Baldwinian ([`Memetic::lamarckian`]);
//! 3. and 4. which offspring, how often: each child with a probability (the frequency);
//! 5. how long: an evaluation cap per local search (the intensity).
//!
//! Frequency and intensity are either fixed ([`Schedule::Fixed`]) or adapted
//! ([`Schedule::Adaptive`]): each is a discrete set of levels with probabilities that move
//! towards the levels that yielded the most relative improvement per evaluation.
//!
//! ```
//! use evolutionary_planning::ga::{Config, Ga, Operators};
//! use evolutionary_planning::memetic::{Memetic, Rosenbrock};
//! use evolutionary_planning::operators::real::{Bounds, Polynomial, Sbx};
//! use evolutionary_planning::operators::{NoRepair, selection::Tournament};
//! use evolutionary_planning::problems::single::rosenbrock;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let bounds = Bounds::uniform(4, -2.0, 2.0);
//! let mut ga = Ga::new(Config { population_size: 10, ..Config::default() }, Operators {
//!     init: bounds.clone(),
//!     selection: Tournament { size: 2 },
//!     crossover: Sbx { bounds: bounds.clone(), eta: 15.0 },
//!     mutation: Polynomial { bounds, eta: 20.0, rate: None },
//!     repair: NoRepair,
//! });
//! let mut memetic = Memetic::adaptive(Rosenbrock::default());
//! let mut objective = |x: &Vec<f64>| rosenbrock(x);
//! let outcome = ga.run_memetic(&mut memetic, &mut objective, &mut Limits::evaluations(20_000), &mut (), &mut SplitMix64::new(1));
//! assert!(outcome.best_cost() < 1e-6);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::evaluator::Evaluator;
use crate::ga::Refine;
use crate::operators::Mutation;
use crate::operators::selection::{Sampling, sample};
use crate::rng::{chance, unit_f64};
use crate::{Budget, Individual, Objective, Observer, Stats};

/// Improves a genome with a bounded number of objective evaluations.
pub trait LocalSearch<G> {
    /// Returns the best individual found starting from `start`, spending at most
    /// `max_evaluations` calls of `evaluate`, which returns `None` once the global budget is
    /// exhausted (stop then).
    fn improve<E, R>(
        &mut self,
        start: Individual<G>,
        max_evaluations: u64,
        evaluate: &mut E,
        rng: &mut R,
    ) -> Individual<G>
    where
        G: Clone,
        E: FnMut(&G) -> Option<f64>,
        R: Rng + ?Sized;
}

/// Random-mutation hill climbing as a local searcher for any genome: mutate, keep if not
/// worse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HillClimb<M> {
    /// The neighbourhood move.
    pub mutation: M,
}

impl<G, M: Mutation<G>> LocalSearch<G> for HillClimb<M> {
    fn improve<E, R>(
        &mut self,
        mut current: Individual<G>,
        max_evaluations: u64,
        evaluate: &mut E,
        rng: &mut R,
    ) -> Individual<G>
    where
        G: Clone,
        E: FnMut(&G) -> Option<f64>,
        R: Rng + ?Sized,
    {
        for _ in 0..max_evaluations {
            let mut candidate = current.genome.clone();
            self.mutation.mutate(&mut candidate, rng);
            let Some(cost) = evaluate(&candidate) else {
                break;
            };

            if cost <= current.cost {
                current = Individual {
                    genome: candidate,
                    cost,
                };
            }
        }

        current
    }
}

/// Rosenbrock's rotating-coordinates method (Rosenbrock 1960), HyGLEAM's local searcher for
/// real parameters: probe along `n` orthonormal directions, growing a step by `alpha` on
/// success and reversing and shrinking it by `beta` on failure; once every direction has had
/// a success and a failure, rotate the directions towards the accumulated progress
/// (Gram-Schmidt) and start over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rosenbrock {
    /// The initial step along every direction.
    pub initial_step: f64,
    /// The expansion factor on success; `3` is Rosenbrock's.
    pub alpha: f64,
    /// The contraction factor on failure; `0.5` is Rosenbrock's.
    pub beta: f64,
    /// Stop when every step is below this.
    pub min_step: f64,
}

impl Default for Rosenbrock {
    fn default() -> Self {
        Self {
            initial_step: 0.1,
            alpha: 3.0,
            beta: 0.5,
            min_step: 1e-12,
        }
    }
}

impl LocalSearch<Vec<f64>> for Rosenbrock {
    fn improve<E, R>(
        &mut self,
        mut current: Individual<Vec<f64>>,
        max_evaluations: u64,
        evaluate: &mut E,
        _rng: &mut R,
    ) -> Individual<Vec<f64>>
    where
        E: FnMut(&Vec<f64>) -> Option<f64>,
        R: Rng + ?Sized,
    {
        let n = current.genome.len();
        let mut directions: Vec<Vec<f64>> = (0..n).map(|i| unit(n, i)).collect();
        let mut steps = alloc::vec![self.initial_step; n];
        let mut progress = alloc::vec![0.0; n];
        let mut succeeded = alloc::vec![false; n];
        let mut failed = alloc::vec![false; n];
        let mut used = 0;
        while used < max_evaluations && steps.iter().any(|s| s.abs() >= self.min_step) {
            for i in 0..n {
                if used >= max_evaluations {
                    break;
                }

                let candidate: Vec<f64> = current
                    .genome
                    .iter()
                    .zip(&directions[i])
                    .map(|(x, d)| x + steps[i] * d)
                    .collect();
                let Some(cost) = evaluate(&candidate) else {
                    return current;
                };
                used += 1;
                if cost <= current.cost {
                    current = Individual {
                        genome: candidate,
                        cost,
                    };
                    progress[i] += steps[i];
                    steps[i] *= self.alpha;
                    succeeded[i] = true;
                } else {
                    steps[i] *= -self.beta;
                    failed[i] = true;
                }
            }

            if succeeded.iter().zip(&failed).all(|(s, f)| *s && *f) {
                directions = rotate(&directions, &progress);
                progress.fill(0.0);
                succeeded.fill(false);
                failed.fill(false);
                steps.fill(self.initial_step);
            }
        }

        current
    }
}

/// The `i`-th unit vector of dimension `n`.
fn unit(n: usize, i: usize) -> Vec<f64> {
    let mut v = alloc::vec![0.0; n];
    v[i] = 1.0;

    v
}

/// Rosenbrock's rotation: `a_i = Σ_{j ≥ i} λ_j d_j`, orthonormalised by Gram-Schmidt; a
/// direction whose `a_i` collapses keeps its old vector (re-orthonormalised).
fn rotate(directions: &[Vec<f64>], progress: &[f64]) -> Vec<Vec<f64>> {
    let n = directions.len();
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut a = alloc::vec![0.0; n];
        for j in i..n {
            for (ak, dk) in a.iter_mut().zip(&directions[j]) {
                *ak += progress[j] * dk;
            }
        }

        let mut v = orthogonalised(&a, &basis);
        if norm(&v) < 1e-12 {
            v = orthogonalised(&directions[i], &basis);
        }

        let length = norm(&v);
        basis.push(if length > 1e-12 {
            v.iter().map(|x| x / length).collect()
        } else {
            directions[i].clone()
        });
    }

    basis
}

/// `a` minus its projections on the orthonormal vectors of `basis`.
fn orthogonalised(a: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut v = a.to_vec();
    for b in basis {
        let dot: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
        v.iter_mut().zip(b).for_each(|(x, y)| *x -= dot * y);
    }

    v
}

/// The Euclidean norm.
fn norm(v: &[f64]) -> f64 {
    libm::sqrt(v.iter().map(|x| x * x).sum())
}

/// A discrete set of parameter levels with adapted probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct Levels<T> {
    /// The level values (evaluation caps or frequencies).
    pub values: Vec<T>,
    /// The probability of each level; sums to one.
    probabilities: Vec<f64>,
    /// The accumulated benefit (relative gain per evaluation) per level since the last
    /// adaptation.
    benefit: Vec<f64>,
    /// The number of uses per level since the last adaptation.
    uses: Vec<u64>,
}

impl<T: Copy> Levels<T> {
    /// Levels with equal initial probabilities.
    ///
    /// # Panics
    ///
    /// Panics if `values` is empty.
    #[must_use]
    pub fn new(values: Vec<T>) -> Self {
        assert!(!values.is_empty(), "at least one level");
        let k = values.len();
        #[expect(clippy::cast_precision_loss, reason = "level counts are tiny")]
        let p = 1.0 / k as f64;

        Self {
            values,
            probabilities: alloc::vec![p; k],
            benefit: alloc::vec![0.0; k],
            uses: alloc::vec![0; k],
        }
    }

    /// The current probabilities.
    #[must_use]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Draws a level index.
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut out = Vec::with_capacity(1);
        sample(&self.probabilities, 1, Sampling::Roulette, rng, &mut out);

        out[0]
    }

    /// Records a use of `level` with `benefit`, and adapts once every level with positive
    /// probability has been used `window` times in total across levels.
    fn record(&mut self, level: usize, benefit: f64, window: u64, floor: f64) {
        self.benefit[level] += benefit;
        self.uses[level] += 1;
        if self.uses.iter().sum::<u64>() < window {
            return;
        }

        // Only levels used in this window carry evidence: they share their combined
        // probability mass in proportion to their mean benefit; unused levels keep theirs.
        #[expect(clippy::cast_precision_loss, reason = "use counts are small")]
        let mean: Vec<f64> = self
            .benefit
            .iter()
            .zip(&self.uses)
            .map(|(b, &u)| if u == 0 { 0.0 } else { b / u as f64 })
            .collect();
        let used_mass: f64 = self
            .probabilities
            .iter()
            .zip(&self.uses)
            .filter(|(_, u)| **u > 0)
            .map(|(p, _)| p)
            .sum();
        let used_benefit: f64 = mean
            .iter()
            .zip(&self.uses)
            .filter(|(_, u)| **u > 0)
            .map(|(m, _)| m)
            .sum();
        if used_benefit > 0.0 {
            // Half old, half new, so one noisy window cannot swing the distribution.
            for ((p, m), &u) in self.probabilities.iter_mut().zip(&mean).zip(&self.uses) {
                if u > 0 {
                    *p = 0.5 * *p + 0.5 * used_mass * m / used_benefit;
                }
            }
        }

        let k = self.probabilities.len();
        #[expect(clippy::cast_precision_loss, reason = "level counts are tiny")]
        let minimum = floor / k as f64;
        for p in &mut self.probabilities {
            *p = p.max(minimum);
        }

        let sum: f64 = self.probabilities.iter().sum();
        self.probabilities.iter_mut().for_each(|p| *p /= sum);
        self.benefit.fill(0.0);
        self.uses.fill(0);
    }
}

/// How often and how long local search runs.
#[derive(Debug, Clone, PartialEq)]
pub enum Schedule {
    /// Each child is improved with `probability`, with at most `evaluations` evaluations.
    Fixed {
        /// The probability of improving a child.
        probability: f64,
        /// The evaluation cap per local search.
        evaluations: u64,
    },
    /// HyGLEAM-style cost-benefit adaptation of both.
    Adaptive(Adaptive),
}

/// The adaptive schedule: frequency and intensity levels whose probabilities follow the
/// relative fitness gain per evaluation they produced.
///
/// The gain of one local search is `(before - after) / max(|before|, 1e-12)`; its benefit
/// is the gain divided by the evaluations spent. After `window` local searches the levels used
/// in that window share their probability mass in proportion to their mean benefit (moving
/// halfway there), unused levels keep theirs, and a floor of `floor / levels` keeps every
/// level alive. The update rule is a reconstruction: Jakob (2021) describes cost-benefit
/// adaptation without formulas.
///
/// Limitation: the benefit is measured per local search, so it reflects intensity directly
/// but frequency only weakly (a frequency level is credited with the searches it triggered,
/// not with what the skipped offspring would have cost); intensity is the parameter that
/// really adapts.
#[derive(Debug, Clone, PartialEq)]
pub struct Adaptive {
    /// The evaluation caps to choose from.
    pub intensity: Levels<u64>,
    /// The improvement probabilities to choose from.
    pub frequency: Levels<f64>,
    /// Local searches between adaptations.
    pub window: u64,
    /// The probability floor, spread over the levels.
    pub floor: f64,
}

impl Default for Adaptive {
    /// Intensities 50, 100, 200, 400, 800 evaluations; frequencies 0.1, 0.25, 0.5, 1;
    /// adaptation every 20 searches with a 10 % floor.
    fn default() -> Self {
        Self {
            intensity: Levels::new(alloc::vec![50, 100, 200, 400, 800]),
            frequency: Levels::new(alloc::vec![0.1, 0.25, 0.5, 1.0]),
            window: 20,
            floor: 0.1,
        }
    }
}

/// A memetic hybrid: a local searcher and its schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct Memetic<L> {
    /// The local searcher.
    pub local: L,
    /// How often and how long it runs.
    pub schedule: Schedule,
    /// Write the improved genome back (Lamarckian); otherwise only its cost is kept
    /// (Baldwinian).
    pub lamarckian: bool,
}

impl<L> Memetic<L> {
    /// A Lamarckian hybrid with the default adaptive schedule.
    #[must_use]
    pub fn adaptive(local: L) -> Self {
        Self {
            local,
            schedule: Schedule::Adaptive(Adaptive::default()),
            lamarckian: true,
        }
    }
}

impl<G, L: LocalSearch<G>> Refine<G> for Memetic<L> {
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
        R: Rng + ?Sized,
    {
        let (apply, cap, levels) = match &self.schedule {
            Schedule::Fixed {
                probability,
                evaluations,
            } => (chance(rng, *probability), *evaluations, None),
            Schedule::Adaptive(adaptive) => {
                let f = adaptive.frequency.draw(rng);
                let i = adaptive.intensity.draw(rng);
                (
                    unit_f64(rng) < adaptive.frequency.values[f],
                    adaptive.intensity.values[i],
                    Some((f, i)),
                )
            }
        };
        if !apply {
            return child;
        }

        let before = child.cost;
        let start_evaluations = evaluator.stats.evaluations;
        let original = (!self.lamarckian).then(|| child.genome.clone());
        let improved = self
            .local
            .improve(child, cap, &mut |g: &G| evaluator.evaluate(g), rng);
        let spent = evaluator.stats.evaluations - start_evaluations;
        if let (Schedule::Adaptive(adaptive), Some((f, i))) = (&mut self.schedule, levels) {
            let gain = (before - improved.cost) / before.abs().max(1e-12);
            #[expect(
                clippy::cast_precision_loss,
                reason = "evaluation counts far below 2^52"
            )]
            let benefit = if spent == 0 || !gain.is_finite() {
                0.0
            } else {
                gain.max(0.0) / spent as f64
            };
            adaptive
                .frequency
                .record(f, benefit, adaptive.window, adaptive.floor);
            adaptive
                .intensity
                .record(i, benefit, adaptive.window, adaptive.floor);
        }

        match original {
            Some(genome) => Individual {
                genome,
                cost: improved.cost,
            },
            None => improved,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ga::{Config, Ga, Operators};
    use crate::operators::NoRepair;
    use crate::operators::real::{Bounds, Polynomial, Sbx};
    use crate::operators::selection::Tournament;
    use crate::problems::single::{ellipsoid, rosenbrock};
    use crate::{Limits, SplitMix64};

    /// Rosenbrock's method alone solves a rotated, badly scaled quadratic and the Rosenbrock
    /// function from a fixed start.
    #[test]
    fn rosenbrock_method_converges() {
        let mut ls = Rosenbrock::default();
        let mut rng = SplitMix64::new(1);
        for (f, start) in [
            (ellipsoid as fn(&[f64]) -> f64, alloc::vec![1.0; 4]),
            (rosenbrock, alloc::vec![-1.2, 1.0]),
        ] {
            let cost = f(&start);
            let mut evaluate = |x: &Vec<f64>| Some(f(x));
            let result = ls.improve(
                Individual {
                    genome: start,
                    cost,
                },
                20_000,
                &mut evaluate,
                &mut rng,
            );
            assert!(result.cost < 1e-8, "{}", result.cost);
        }
    }

    /// The memetic GA beats the plain GA on Rosenbrock at the same budget, and the adaptive
    /// schedule keeps valid probability distributions.
    #[test]
    fn memetic_beats_plain_ga() {
        let bounds = Bounds::uniform(5, -2.0, 2.0);
        let make = || {
            Ga::new(
                Config {
                    population_size: 20,
                    elitism: 1,
                    ..Config::default()
                },
                Operators {
                    init: bounds.clone(),
                    selection: Tournament { size: 2 },
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
        };
        let mut objective = |x: &Vec<f64>| rosenbrock(x);
        let plain = make().run(
            &mut objective,
            &mut Limits::evaluations(30_000),
            &mut (),
            &mut SplitMix64::new(3),
        );
        let mut memetic = Memetic::adaptive(Rosenbrock::default());
        let hybrid = make().run_memetic(
            &mut memetic,
            &mut objective,
            &mut Limits::evaluations(30_000),
            &mut (),
            &mut SplitMix64::new(3),
        );

        assert_eq!(hybrid.stats.evaluations, 30_000);
        assert!(
            hybrid.best_cost() < plain.best_cost(),
            "{} vs {}",
            hybrid.best_cost(),
            plain.best_cost()
        );
        assert!(hybrid.best_cost() < 1e-6, "{}", hybrid.best_cost());
        if let Schedule::Adaptive(a) = &memetic.schedule {
            for p in [a.intensity.probabilities(), a.frequency.probabilities()] {
                assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                assert!(p.iter().all(|&x| x > 0.0));
            }
        }
    }

    /// A Baldwinian hybrid keeps the genome but takes the improved cost.
    #[test]
    fn baldwinian_keeps_genome() {
        let mut memetic = Memetic {
            local: Rosenbrock::default(),
            schedule: Schedule::Fixed {
                probability: 1.0,
                evaluations: 200,
            },
            lamarckian: false,
        };
        let mut objective = |x: &Vec<f64>| x.iter().map(|v| v * v).sum::<f64>();
        let mut budget = Limits::unlimited();
        let mut observer = ();
        let mut evaluator = Evaluator::new(&mut objective, &mut budget, &mut observer);
        let child = Individual {
            genome: alloc::vec![1.0, 1.0],
            cost: 2.0,
        };
        let refined = memetic.refine(child, &mut evaluator, &mut SplitMix64::new(1));

        assert_eq!(refined.genome, [1.0, 1.0]);
        assert!(refined.cost < 2.0);
    }
}
