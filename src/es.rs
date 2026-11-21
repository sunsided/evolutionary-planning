//! Evolution strategies on real vectors: the (μ/ρ,λ)- and (μ/ρ+λ)-ES with self-adaptive step
//! sizes (Schwefel; Beyer and Schwefel 2002), and the (1+1)-ES with the 1/5th success rule
//! (Rechenberg).
//!
//! ```
//! use evolutionary_planning::es::{Config, Es};
//! use evolutionary_planning::operators::real::Bounds;
//! use evolutionary_planning::problems::single::sphere;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut es = Es::new(Config::default(), Bounds::uniform(5, -3.0, 3.0));
//! let mut objective = |x: &Vec<f64>| sphere(x);
//! let outcome = es.run(&mut objective, &mut Limits::evaluations(20_000), &mut (), &mut SplitMix64::new(1));
//! assert!(outcome.best_cost() < 1e-8);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::evaluator::Evaluator;
use crate::operators::Initializer;
use crate::rng::{below, distinct_indices, standard_normal};
use crate::{Budget, Individual, Objective, Observer, Outcome, Stats, Stop};

/// Which individuals compete for survival.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Survival {
    /// (μ,λ): the μ best of the λ offspring; parents die. Needs `λ >= μ`.
    Comma,
    /// (μ+λ): the μ best of parents and offspring together (elitist).
    Plus,
}

/// How many step sizes each individual carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepSizes {
    /// One σ, mutated with learning rate `τ = 1/sqrt(n)`.
    Global,
    /// One σ per coordinate, with `τ' = 1/sqrt(2n)` (global factor) and
    /// `τ = 1/sqrt(2 sqrt(n))` (per coordinate).
    PerCoordinate,
}

/// How the object variables of the ρ parents are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recombination {
    /// The centroid of the ρ parents (step sizes: arithmetic mean).
    Intermediate,
    /// Each coordinate (and its step size) from a uniformly chosen parent.
    Discrete,
}

/// Parameters of an [`Es`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The number of parents μ, at least `1`.
    pub mu: usize,
    /// The number of parents recombined per offspring, `1..=mu`.
    pub rho: usize,
    /// The number of offspring λ per generation; at least μ for [`Survival::Comma`].
    pub lambda: usize,
    /// Comma or plus selection.
    pub survival: Survival,
    /// Global or per-coordinate step sizes.
    pub step_sizes: StepSizes,
    /// Intermediate or discrete recombination.
    pub recombination: Recombination,
    /// The initial step size, as a fraction of each coordinate's bound width.
    pub initial_sigma: f64,
    /// Step sizes never fall below this (absolute); prevents premature collapse to zero.
    pub min_sigma: f64,
}

impl Default for Config {
    /// A (15/2,100)-ES with per-coordinate step sizes, intermediate recombination, initial σ
    /// a third of the bound width.
    fn default() -> Self {
        Self {
            mu: 15,
            rho: 2,
            lambda: 100,
            survival: Survival::Comma,
            step_sizes: StepSizes::PerCoordinate,
            recombination: Recombination::Intermediate,
            initial_sigma: 0.3,
            min_sigma: 1e-300,
        }
    }
}

/// One ES individual: object variables, step sizes and cost.
#[derive(Debug, Clone, PartialEq)]
struct Member {
    /// The object variables.
    x: Vec<f64>,
    /// One step size (global) or one per coordinate.
    sigma: Vec<f64>,
    /// The cost of `x`.
    cost: f64,
}

/// A self-adaptive evolution strategy over `Vec<f64>` within box bounds (offspring leaving
/// the box are reflected back in).
#[derive(Debug, Clone, PartialEq)]
pub struct Es {
    /// The run parameters.
    pub config: Config,
    /// The search box: initial parents are uniform in it, offspring are reflected into it.
    pub bounds: crate::operators::real::Bounds,
}

impl Es {
    /// Creates an ES.
    #[must_use]
    pub fn new(config: Config, bounds: crate::operators::real::Bounds) -> Self {
        Self { config, bounds }
    }

    /// Runs until the budget stops it.
    ///
    /// # Panics
    ///
    /// Panics if `mu`, `rho` or `lambda` is `0`, `rho > mu`, or `lambda < mu` with comma
    /// selection.
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
        self.validate();
        let mut evaluator = Evaluator::new(objective, budget, observer);
        let Some(mut parents) = self.initial_parents(&mut evaluator, rng) else {
            return evaluator.outcome(Vec::new(), Stop::Budget);
        };

        loop {
            let mut offspring = Vec::with_capacity(self.config.lambda);
            for _ in 0..self.config.lambda {
                let child = self.offspring(&parents, rng);
                let Some(cost) = evaluator.evaluate(&child.x) else {
                    return evaluator.outcome(population(parents), Stop::Budget);
                };
                offspring.push(Member { cost, ..child });
            }

            if self.config.survival == Survival::Plus {
                offspring.extend(parents);
            }

            offspring.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            offspring.truncate(self.config.mu);
            parents = offspring;
            evaluator.end_generation(parents[0].cost);
        }
    }

    /// Checks the configuration.
    fn validate(&self) {
        let c = &self.config;
        assert!(
            c.mu > 0 && c.lambda > 0 && c.rho > 0,
            "mu, rho and lambda must be positive"
        );
        assert!(c.rho <= c.mu, "rho must not exceed mu");
        assert!(
            c.survival == Survival::Plus || c.lambda >= c.mu,
            "comma selection needs lambda >= mu"
        );
    }

    /// μ uniform parents with the initial step sizes; `None` if the budget ran out.
    fn initial_parents<O, B, Obs, R>(
        &mut self,
        evaluator: &mut Evaluator<'_, Vec<f64>, O, B, Obs>,
        rng: &mut R,
    ) -> Option<Vec<Member>>
    where
        O: Objective<Vec<f64>> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let n = self.bounds.dimension();
        let sigma: Vec<f64> = match self.config.step_sizes {
            StepSizes::Global => {
                let mean_width = (0..n).map(|i| self.bounds.width(i)).sum::<f64>() / dimension(n);
                alloc::vec![self.config.initial_sigma * mean_width; 1]
            }
            StepSizes::PerCoordinate => (0..n)
                .map(|i| self.config.initial_sigma * self.bounds.width(i))
                .collect(),
        };

        let mut parents = Vec::with_capacity(self.config.mu);
        for _ in 0..self.config.mu {
            let x = self.bounds.initialize(rng);
            let cost = evaluator.evaluate(&x)?;
            parents.push(Member {
                x,
                sigma: sigma.clone(),
                cost,
            });
        }

        Some(parents)
    }

    /// Recombines ρ parents and mutates step sizes, then object variables.
    fn offspring<R: Rng + ?Sized>(&self, parents: &[Member], rng: &mut R) -> Member {
        let chosen = distinct_indices(rng, parents.len(), self.config.rho, None);
        let mut child = self.recombine(parents, &chosen, rng);
        self.mutate_sigma(&mut child.sigma, rng);
        for (i, x) in child.x.iter_mut().enumerate() {
            let sigma = if child.sigma.len() == 1 {
                child.sigma[0]
            } else {
                child.sigma[i]
            };
            *x += sigma * standard_normal(rng);
        }

        let mut repair = crate::operators::real::Reflect(self.bounds.clone());
        crate::operators::Repair::repair(&mut repair, &mut child.x, rng);
        child
    }

    /// Recombines the parents at `chosen` into an unevaluated child.
    fn recombine<R: Rng + ?Sized>(
        &self,
        parents: &[Member],
        chosen: &[usize],
        rng: &mut R,
    ) -> Member {
        let n = parents[0].x.len();
        let k = parents[0].sigma.len();
        let mut x = alloc::vec![0.0; n];
        let mut sigma = alloc::vec![0.0; k];
        match self.config.recombination {
            Recombination::Intermediate => {
                #[expect(clippy::cast_precision_loss, reason = "rho is tiny")]
                let weight = 1.0 / chosen.len() as f64;
                for &p in chosen {
                    x.iter_mut()
                        .zip(&parents[p].x)
                        .for_each(|(a, b)| *a += weight * b);
                    sigma
                        .iter_mut()
                        .zip(&parents[p].sigma)
                        .for_each(|(a, b)| *a += weight * b);
                }
            }
            Recombination::Discrete => {
                for (i, xi) in x.iter_mut().enumerate() {
                    *xi = parents[chosen[below(rng, chosen.len())]].x[i];
                }
                for (j, s) in sigma.iter_mut().enumerate() {
                    *s = parents[chosen[below(rng, chosen.len())]].sigma[j];
                }
            }
        }

        Member {
            x,
            sigma,
            cost: f64::INFINITY,
        }
    }

    /// Log-normal self-adaptation of the step sizes.
    fn mutate_sigma<R: Rng + ?Sized>(&self, sigma: &mut [f64], rng: &mut R) {
        let n = dimension(self.bounds.dimension());
        let floor = self.config.min_sigma;
        if sigma.len() == 1 {
            let tau = 1.0 / libm::sqrt(n);
            sigma[0] = (sigma[0] * libm::exp(tau * standard_normal(rng))).max(floor);
            return;
        }

        let tau_global = 1.0 / libm::sqrt(2.0 * n);
        let tau_local = 1.0 / libm::sqrt(2.0 * libm::sqrt(n));
        let global = tau_global * standard_normal(rng);
        for s in sigma.iter_mut() {
            *s = (*s * libm::exp(global + tau_local * standard_normal(rng))).max(floor);
        }
    }
}

/// The dimension as a float, at least `1`.
fn dimension(n: usize) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
    let n = n.max(1) as f64;

    n
}

/// Converts ES members into outcome individuals.
fn population(members: Vec<Member>) -> Vec<Individual<Vec<f64>>> {
    members
        .into_iter()
        .map(|m| Individual {
            genome: m.x,
            cost: m.cost,
        })
        .collect()
}

/// The (1+1)-ES with the 1/5th success rule: one parent, one Gaussian offspring per
/// iteration, accepted if not worse; σ grows by `exp(1/3)` on success and shrinks by
/// `exp(-1/12)` on failure, which keeps it stationary at a success rate of exactly 1/5.
///
/// ```
/// use evolutionary_planning::es::OnePlusOne;
/// use evolutionary_planning::problems::single::sphere;
/// use evolutionary_planning::{Limits, SplitMix64};
///
/// let mut es = OnePlusOne { sigma: 1.0, min_sigma: 1e-300 };
/// let mut objective = |x: &Vec<f64>| sphere(x);
/// let outcome = es.run(vec![2.0; 5], &mut objective, &mut Limits::evaluations(3_000), &mut (), &mut SplitMix64::new(1));
/// assert!(outcome.best_cost() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnePlusOne {
    /// The initial step size (absolute).
    pub sigma: f64,
    /// Step sizes never fall below this.
    pub min_sigma: f64,
}

impl OnePlusOne {
    /// Runs from `start` until the budget stops it; one iteration is one generation.
    pub fn run<O, B, Obs, R>(
        &mut self,
        start: Vec<f64>,
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
        let mut evaluator = Evaluator::new(objective, budget, observer);
        let Some(mut current) = evaluator.individual(start) else {
            return evaluator.outcome(Vec::new(), Stop::Budget);
        };

        let mut sigma = self.sigma;
        loop {
            let candidate: Vec<f64> = current
                .genome
                .iter()
                .map(|x| x + sigma * standard_normal(rng))
                .collect();
            let Some(cost) = evaluator.evaluate(&candidate) else {
                break;
            };

            if cost <= current.cost {
                current = Individual {
                    genome: candidate,
                    cost,
                };
                sigma *= libm::exp(1.0 / 3.0);
            } else {
                sigma = (sigma * libm::exp(-1.0 / 12.0)).max(self.min_sigma);
            }

            evaluator.end_generation(current.cost);
        }

        evaluator.outcome(alloc::vec![current], Stop::Budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::real::Bounds;
    use crate::problems::single::{ellipsoid, rosenbrock, sphere};
    use crate::{Limits, SplitMix64};

    /// Every survival, step-size and recombination combination converges on the sphere.
    #[test]
    fn variants_converge_on_sphere() {
        for survival in [Survival::Comma, Survival::Plus] {
            for step_sizes in [StepSizes::Global, StepSizes::PerCoordinate] {
                for recombination in [Recombination::Intermediate, Recombination::Discrete] {
                    let config = Config {
                        survival,
                        step_sizes,
                        recombination,
                        ..Config::default()
                    };
                    let mut es = Es::new(config, Bounds::uniform(4, -5.0, 5.0));
                    let mut objective = |x: &Vec<f64>| sphere(x);
                    let outcome = es.run(
                        &mut objective,
                        &mut Limits::evaluations(30_000),
                        &mut (),
                        &mut SplitMix64::new(3),
                    );
                    assert!(
                        outcome.best_cost() < 1e-6,
                        "{config:?}: {}",
                        outcome.best_cost()
                    );
                }
            }
        }
    }

    /// Per-coordinate step sizes cope with a badly scaled ellipsoid.
    #[test]
    fn per_coordinate_on_ellipsoid() {
        let mut es = Es::new(Config::default(), Bounds::uniform(5, -3.0, 3.0));
        let mut objective = |x: &Vec<f64>| ellipsoid(x);
        let outcome = es.run(
            &mut objective,
            &mut Limits::evaluations(100_000),
            &mut (),
            &mut SplitMix64::new(1),
        );

        assert!(outcome.best_cost() < 1e-6, "{}", outcome.best_cost());
    }

    /// The (1+1)-ES converges linearly on the sphere and makes progress on Rosenbrock.
    #[test]
    fn one_plus_one() {
        let mut es = OnePlusOne {
            sigma: 1.0,
            min_sigma: 1e-300,
        };
        let mut objective = |x: &Vec<f64>| sphere(x);
        let outcome = es.run(
            alloc::vec![3.0; 10],
            &mut objective,
            &mut Limits::evaluations(5_000),
            &mut (),
            &mut SplitMix64::new(2),
        );
        assert!(outcome.best_cost() < 1e-10, "{}", outcome.best_cost());

        let mut objective = |x: &Vec<f64>| rosenbrock(x);
        let outcome = es.run(
            alloc::vec![0.0; 2],
            &mut objective,
            &mut Limits::evaluations(20_000),
            &mut (),
            &mut SplitMix64::new(2),
        );
        assert!(outcome.best_cost() < 1e-6, "{}", outcome.best_cost());
    }

    /// Same seed, same run; the budget is exact.
    #[test]
    fn deterministic() {
        let run = || {
            let mut es = Es::new(Config::default(), Bounds::uniform(3, -1.0, 1.0));
            let mut objective = |x: &Vec<f64>| sphere(x);
            es.run(
                &mut objective,
                &mut Limits::evaluations(777),
                &mut (),
                &mut SplitMix64::new(9),
            )
        };

        assert_eq!(run(), run());
        assert_eq!(run().stats.evaluations, 777);
    }
}
