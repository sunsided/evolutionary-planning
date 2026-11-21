//! Online trajectory optimisation with warm-started CMA-ES, after the online motion planner
//! of Yue et al. (2023): a trajectory is a control sequence of `steps × controls` reals that
//! the caller's objective rolls out through its dynamics, and every control period the plan
//! is re-optimised under a small evaluation budget, starting from
//!
//! 1. the previous solution shifted by one step, keeping σ and the covariance (also shifted),
//!    when there is one (receding horizon); otherwise
//! 2. a stored solution of a nearby task from a [`WarmStartLibrary`] (Yue et al.'s pre-motion
//!    library, within a radius of the task descriptor); otherwise
//! 3. a cold start at the centre of the bounds (or zero).
//!
//! The paper warm-starts differential evolution; here the same library seeds CMA-ES, whose
//! mean, step size and covariance carry over between replans (see the paper table).
//!
//! ```
//! use evolutionary_planning::trajectory::{Config, OnlineOptimizer};
//! use evolutionary_planning::SplitMix64;
//!
//! // A 1D point mass with acceleration controls; drive it from x = 1 to rest at 0.
//! let rollout = |state: (f64, f64), controls: &[f64]| {
//!     let (mut x, mut v, mut cost) = (state.0, state.1, 0.0);
//!     for &a in controls {
//!         v += 0.1 * a;
//!         x += 0.1 * v;
//!         cost += x * x + 0.1 * v * v + 0.01 * a * a;
//!     }
//!     cost
//! };
//! let mut planner = OnlineOptimizer::new(Config { steps: 15, controls: 1, ..Config::default() });
//! let mut rng = SplitMix64::new(1);
//! let mut state = (1.0, 0.0);
//! for _ in 0..60 {
//!     let plan = planner.replan(&mut |u: &Vec<f64>| rollout(state, u), None, &mut rng);
//!     let a = plan.first()[0];
//!     state.1 += 0.1 * a;
//!     state.0 += 0.1 * state.1;
//! }
//! assert!(state.0.abs() < 0.1);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::cmaes::{CmaEs, Config as CmaConfig};
use crate::de::WarmStartLibrary;
use crate::operators::real::Bounds;
use crate::{Limits, Objective};

/// Parameters of an [`OnlineOptimizer`].
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The number of control steps in a plan, at least `1`.
    pub steps: usize,
    /// The number of controls per step, at least `1`.
    pub controls: usize,
    /// The step size of a cold or library start.
    pub sigma: f64,
    /// The smallest step size a warm start begins with, so the new tail of a shifted plan
    /// can still move.
    pub min_sigma: f64,
    /// Objective evaluations per replan.
    pub evaluations: u64,
    /// CMA-ES population size; `None` for the default.
    pub lambda: Option<usize>,
    /// Per-control bounds `(low, high)`, applied to every step by CMA-ES's boundary penalty;
    /// `None` is unbounded.
    pub bounds: Option<Vec<(f64, f64)>>,
    /// The radius within which a library entry warm-starts a new task.
    pub library_radius: f64,
}

impl Default for Config {
    /// 20 steps of one control, `σ = 0.5`, `σ_min = 0.05`, 300 evaluations per replan, no
    /// bounds, library radius `0.05` (the paper's threshold).
    fn default() -> Self {
        Self {
            steps: 20,
            controls: 1,
            sigma: 0.5,
            min_sigma: 0.05,
            evaluations: 300,
            lambda: None,
            bounds: None,
            library_radius: 0.05,
        }
    }
}

/// Where a replan started from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmStart {
    /// From scratch.
    Cold,
    /// From the previous plan shifted by one step.
    Shifted,
    /// From a library entry of a nearby task.
    Library,
}

/// The result of one replan.
#[derive(Debug, Clone, PartialEq)]
pub struct Replan {
    /// The best control sequence, `steps × controls` values, step-major.
    pub controls: Vec<f64>,
    /// Its cost.
    pub cost: f64,
    /// How the search was started.
    pub warm_start: WarmStart,
    /// The number of controls per step.
    per_step: usize,
}

impl Replan {
    /// The controls of the first step: what to execute now.
    #[must_use]
    pub fn first(&self) -> &[f64] {
        &self.controls[..self.per_step]
    }
}

/// The state carried from one replan to the next.
#[derive(Debug, Clone, PartialEq)]
struct Carry {
    /// The best controls of the last replan.
    controls: Vec<f64>,
    /// The final step size.
    sigma: f64,
    /// The final covariance, row-major.
    covariance: Vec<f64>,
}

/// A receding-horizon trajectory optimiser.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineOptimizer {
    /// The parameters.
    pub config: Config,
    /// Solutions of solved tasks, by descriptor.
    pub library: WarmStartLibrary,
    /// The previous replan's state, if any.
    carry: Option<Carry>,
}

impl OnlineOptimizer {
    /// A fresh optimiser with an empty library.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self {
            config,
            library: WarmStartLibrary::new(),
            carry: None,
        }
    }

    /// Starts a new task: the next replan does not shift the previous plan (it may use the
    /// library).
    pub fn reset(&mut self) {
        self.carry = None;
    }

    /// Stores the last plan in the library under `descriptor` (for example the task's target),
    /// for warm starts of nearby tasks after a [`OnlineOptimizer::reset`].
    pub fn remember(&mut self, descriptor: Vec<f64>) {
        if let Some(carry) = &self.carry {
            self.library.insert(descriptor, carry.controls.clone());
        }
    }

    /// Re-optimises the plan under `objective` (which rolls a control sequence out from the
    /// current state), consulting the library with `descriptor` on a fresh task.
    ///
    /// # Panics
    ///
    /// Panics if `steps` or `controls` is `0`, or the bounds do not have one pair per control.
    pub fn replan<O, R>(
        &mut self,
        objective: &mut O,
        descriptor: Option<&[f64]>,
        rng: &mut R,
    ) -> Replan
    where
        O: Objective<Vec<f64>> + ?Sized,
        R: Rng + ?Sized,
    {
        let c = &self.config;
        assert!(
            c.steps > 0 && c.controls > 0,
            "a plan needs at least one step and one control"
        );
        let n = c.steps * c.controls;
        let cma_config = CmaConfig {
            lambda: c.lambda,
            bounds: self.bounds(),
            ..CmaConfig::default()
        };
        let (mut cma, warm_start) = self.start(n, cma_config, descriptor);
        let outcome = cma.minimize(
            objective,
            &mut Limits::evaluations(self.config.evaluations),
            &mut (),
            rng,
        );
        let (controls, cost) = outcome.best.map_or_else(
            || (cma.mean().to_vec(), f64::INFINITY),
            |b| (b.genome, b.cost),
        );
        let (sigma, covariance) = carried_distribution(&cma);
        self.carry = Some(Carry {
            controls: controls.clone(),
            sigma,
            covariance,
        });

        Replan {
            controls,
            cost,
            warm_start,
            per_step: self.config.controls,
        }
    }

    /// The flattened bounds of a whole plan, if any.
    fn bounds(&self) -> Option<Bounds> {
        let per_control = self.config.bounds.as_ref()?;
        assert_eq!(
            per_control.len(),
            self.config.controls,
            "one bound pair per control"
        );
        let (lower, upper) = (0..self.config.steps)
            .flat_map(|_| per_control.iter().copied())
            .unzip();

        Some(Bounds::new(lower, upper))
    }

    /// Chooses the warm start and builds the CMA-ES state.
    fn start(&self, n: usize, config: CmaConfig, descriptor: Option<&[f64]>) -> (CmaEs, WarmStart) {
        let c = &self.config;
        if let Some(carry) = self
            .carry
            .as_ref()
            .filter(|carry| carry.controls.len() == n)
        {
            let mean = shift_controls(&carry.controls, c.controls);
            let covariance = shift_covariance(&carry.covariance, n, c.controls);
            let sigma = carry.sigma.max(c.min_sigma);
            return (
                CmaEs::with_covariance(mean, sigma, covariance, config),
                WarmStart::Shifted,
            );
        }

        let library_hit = descriptor
            .and_then(|d| self.library.nearest(d, c.library_radius))
            .filter(|(_, solution)| solution.len() == n);
        if let Some((_, solution)) = library_hit {
            return (
                CmaEs::new(solution.to_vec(), c.sigma, config),
                WarmStart::Library,
            );
        }

        let mean = config
            .bounds
            .as_ref()
            .map_or_else(|| alloc::vec![0.0; n], Bounds::center);
        (CmaEs::new(mean, c.sigma, config), WarmStart::Cold)
    }
}

/// The step size and covariance to carry into the next replan: `C` scaled to unit mean
/// variance (with σ scaled inversely, the same distribution), so that the σ floor of the warm
/// start bounds the actual spread; reset to `I` when `C` has become ill-conditioned (above
/// `1e12`), which happens over long runs once the plan has converged and the task changes.
fn carried_distribution(cma: &CmaEs) -> (f64, Vec<f64>) {
    let n = cma.mean().len();
    let covariance = cma.covariance();
    #[expect(clippy::cast_precision_loss, reason = "plan sizes are far below 2^52")]
    let mean_variance = (0..n).map(|i| covariance[i * n + i]).sum::<f64>() / n as f64;
    if !(mean_variance > 0.0 && mean_variance.is_finite()) || cma.condition() > 1e12 {
        let mut identity = alloc::vec![0.0; n * n];
        (0..n).for_each(|i| identity[i * n + i] = 1.0);
        return (cma.sigma(), identity);
    }

    let scaled = covariance.iter().map(|c| c / mean_variance).collect();

    (cma.sigma() * libm::sqrt(mean_variance), scaled)
}

/// Drops the first step and repeats the last one: the receding-horizon shift of a plan.
fn shift_controls(controls: &[f64], per_step: usize) -> Vec<f64> {
    let mut shifted = controls[per_step..].to_vec();
    shifted.extend_from_slice(&controls[controls.len() - per_step..]);

    shifted
}

/// Shifts a covariance over plans by one step: the block of steps `1..` moves to `0..`, and
/// the new last step gets an uncorrelated block with the old last step's variances.
fn shift_covariance(covariance: &[f64], n: usize, per_step: usize) -> Vec<f64> {
    let mut shifted = alloc::vec![0.0; n * n];
    let kept = n - per_step;
    for i in 0..kept {
        for j in 0..kept {
            shifted[i * n + j] = covariance[(i + per_step) * n + j + per_step];
        }
    }

    for i in kept..n {
        shifted[i * n + i] = covariance[i * n + i];
    }

    shifted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMix64;

    /// The cost of a control sequence for a 1D point mass heading for `target`.
    fn point_mass(state: (f64, f64), target: f64, controls: &[f64]) -> f64 {
        let (mut x, mut v, mut cost) = (state.0, state.1, 0.0);
        for &a in controls {
            v += 0.1 * a;
            x += 0.1 * v;
            cost += (x - target) * (x - target) + 0.1 * v * v + 0.01 * a * a;
        }

        cost
    }

    /// Runs a closed loop of `periods` replans; returns the accumulated closed-loop cost.
    fn closed_loop(planner: &mut OnlineOptimizer, warm: bool, periods: usize, seed: u64) -> f64 {
        let mut rng = SplitMix64::new(seed);
        let mut state = (1.0, 0.0);
        let mut total = 0.0;
        for _ in 0..periods {
            if !warm {
                planner.reset();
            }

            let plan = planner.replan(
                &mut |u: &Vec<f64>| point_mass(state, 0.0, u),
                None,
                &mut rng,
            );
            let a = plan.first()[0];
            state.1 += 0.1 * a;
            state.0 += 0.1 * state.1;
            total += state.0 * state.0 + 0.1 * state.1 * state.1 + 0.01 * a * a;
        }

        total
    }

    /// With a tight budget, shifting the previous plan beats re-planning from scratch.
    #[test]
    fn shifted_warm_start_beats_cold() {
        let config = Config {
            steps: 20,
            evaluations: 60,
            ..Config::default()
        };
        let warm = closed_loop(&mut OnlineOptimizer::new(config.clone()), true, 50, 3);
        let cold = closed_loop(&mut OnlineOptimizer::new(config), false, 50, 3);

        assert!(warm < cold, "warm {warm} vs cold {cold}");
    }

    /// A library entry of a nearby task warm-starts a new task to a better plan.
    #[test]
    fn library_warm_start() {
        let config = Config {
            steps: 15,
            evaluations: 3_000,
            ..Config::default()
        };
        let mut planner = OnlineOptimizer::new(config);
        let mut rng = SplitMix64::new(4);
        planner.replan(
            &mut |u: &Vec<f64>| point_mass((0.0, 0.0), 1.0, u),
            Some(&[1.0]),
            &mut rng,
        );
        planner.remember(alloc::vec![1.0]);

        planner.reset();
        planner.config.evaluations = 50;
        let warm = planner.replan(
            &mut |u: &Vec<f64>| point_mass((0.0, 0.0), 1.02, u),
            Some(&[1.02]),
            &mut rng,
        );
        planner.reset();
        let cold = planner.replan(
            &mut |u: &Vec<f64>| point_mass((0.0, 0.0), 1.02, u),
            Some(&[3.0]),
            &mut rng,
        );

        assert_eq!(warm.warm_start, WarmStart::Library);
        assert_eq!(cold.warm_start, WarmStart::Cold);
        assert!(warm.cost < cold.cost, "{} vs {}", warm.cost, cold.cost);
    }

    /// Over a thousand periods with the target moving halfway, the warm-started loop
    /// re-converges: the carried distribution never collapses.
    #[test]
    fn long_run_with_target_change() {
        let config = Config {
            steps: 15,
            evaluations: 60,
            ..Config::default()
        };
        let mut planner = OnlineOptimizer::new(config);
        let mut rng = SplitMix64::new(6);
        let mut state = (0.0, 0.0);
        for period in 0..1_000 {
            let target = if period < 500 { 0.5 } else { -0.5 };
            let plan = planner.replan(
                &mut |u: &Vec<f64>| point_mass(state, target, u),
                None,
                &mut rng,
            );
            let a = plan.first()[0];
            state.1 += 0.1 * a;
            state.0 += 0.1 * state.1;
            if period == 499 {
                assert!((state.0 - 0.5).abs() < 0.05, "first target: {state:?}");
            }
        }

        assert!(
            (state.0 + 0.5).abs() < 0.05 && state.1.abs() < 0.1,
            "second target: {state:?}"
        );
    }

    /// The shifts move blocks by one step.
    #[test]
    fn shifts() {
        assert_eq!(
            shift_controls(&[1.0, 2.0, 3.0, 4.0], 2),
            [3.0, 4.0, 3.0, 4.0]
        );
        let c = [1.0, 0.1, 0.1, 2.0];
        assert_eq!(shift_covariance(&c, 2, 1), [2.0, 0.0, 0.0, 2.0]);
    }

    /// Bounds keep the executed controls inside.
    #[test]
    fn bounded_controls() {
        let config = Config {
            steps: 10,
            bounds: Some(alloc::vec![(-0.2, 0.2)]),
            ..Config::default()
        };
        let mut planner = OnlineOptimizer::new(config);
        let plan = planner.replan(
            &mut |u: &Vec<f64>| point_mass((1.0, 0.0), 0.0, u),
            None,
            &mut SplitMix64::new(1),
        );

        assert!(plan.controls.iter().all(|a| (-0.2..=0.2).contains(a)));
    }
}
