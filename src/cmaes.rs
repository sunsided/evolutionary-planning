//! CMA-ES, the covariance matrix adaptation evolution strategy, exactly as in Hansen's
//! tutorial (arXiv:1604.00772): Figure 6 with the default parameters of Table 1, including
//! the negative ("active") recombination weights, `h_σ` and `δ(h_σ)`, the rescaling (46) of
//! negatively weighted steps, the lazy eigendecomposition of Appendix B.2 and the
//! termination criteria of Appendix B.3. IPOP restarts (Auger and Hansen 2005) with growing
//! population size are in [`Ipop`].
//!
//! ```mermaid
//! graph LR
//!     ask["ask: z ~ N(0, I), y = B D z, x = m + σ y"] --> eval[evaluate and rank]
//!     eval --> mean["m += c_m σ ⟨y⟩_w"]
//!     mean --> psigma["p_σ, σ (43, 44)"]
//!     psigma --> pc["h_σ, p_c (45)"]
//!     pc --> cov["C (46, 47)"]
//!     cov --> eigen["every k generations: C = B D² Bᵀ"]
//!     eigen --> stop{B.3 criteria}
//!     stop -->|continue| ask
//! ```
//!
//! Two interfaces: ask-and-tell ([`CmaEs::ask`], [`CmaEs::tell`]) for callers that evaluate
//! a batch themselves (for example in parallel), and [`CmaEs::minimize`] with a budget.
//!
//! ```
//! use evolutionary_planning::cmaes::{CmaEs, Config};
//! use evolutionary_planning::problems::single::rosenbrock;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut cma = CmaEs::new(vec![0.0; 4], 0.5, Config::default());
//! let mut objective = |x: &Vec<f64>| rosenbrock(x);
//! let outcome = cma.minimize(&mut objective, &mut Limits::evaluations(20_000).with_target(1e-10), &mut (), &mut SplitMix64::new(1));
//! assert!(outcome.best_cost() <= 1e-10);
//! ```

mod eigen;
mod params;
mod termination;

use alloc::vec::Vec;

use rand_core::Rng;

pub use params::Parameters;
use termination::History;
pub use termination::Termination;

use crate::evaluator::Evaluator;
use crate::operators::real::Bounds;
use crate::rng::{standard_normal, uniform};
use crate::{
    Budget, Event, Individual, Objective, Observer, Outcome, Stats, Stop, ranking, sanitize,
};

/// Evaluated candidates of one generation (repaired points with their objective values).
type Population = Vec<Individual<Vec<f64>>>;

/// User-settable options of a [`CmaEs`] run; everything else follows Table 1.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The population size λ; `None` means the default `4 + ⌊3 ln n⌋`. Increasing it improves
    /// global search at the price of speed; decreasing it is not recommended.
    pub lambda: Option<usize>,
    /// TolFun: stop when recent best costs and the current generation's costs lie within this
    /// range. Default `1e-12`.
    pub tol_fun: f64,
    /// TolX: stop when all standard deviations and `σ p_c` fall below this. `None` means
    /// `1e-12` times the initial σ.
    pub tol_x: Option<f64>,
    /// TolXUp: stop when `σ max(diag D)` grew by more than this factor. Default `1e4`.
    pub tol_x_up: f64,
    /// ConditionCov: stop when the condition number of C exceeds this. Default `1e14`.
    pub max_condition: f64,
    /// Box bounds, handled by the penalisation (63) of Appendix B.5 in
    /// [`CmaEs::minimize`]: the objective is evaluated at the closest point in the box, and
    /// `penalty · ‖x - x_repaired‖²` is added for ranking only. `None` is unbounded.
    pub bounds: Option<Bounds>,
    /// The penalty factor α of (63); choose it so that penalty and cost differences have a
    /// similar magnitude. Default `1`.
    pub penalty: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lambda: None,
            tol_fun: 1e-12,
            tol_x: None,
            tol_x_up: 1e4,
            max_condition: 1e14,
            bounds: None,
            penalty: 1.0,
        }
    }
}

/// The state of one CMA-ES run.
#[derive(Debug, Clone, PartialEq)]
pub struct CmaEs {
    /// The options.
    config: Config,
    /// The strategy parameters of Table 1.
    params: Parameters,
    /// The distribution mean `m`.
    mean: Vec<f64>,
    /// The step size `σ`.
    sigma: f64,
    /// The covariance matrix `C`, row-major `n × n`, symmetric.
    covariance: Vec<f64>,
    /// The eigenvectors of `C` as columns of a row-major matrix `B`, from the last
    /// decomposition.
    basis: Vec<f64>,
    /// The square roots of the eigenvalues of `C`, the diagonal of `D`.
    scales: Vec<f64>,
    /// The evolution path of the step size, `p_σ`.
    path_sigma: Vec<f64>,
    /// The evolution path of the rank-one update, `p_c`.
    path_c: Vec<f64>,
    /// Completed generations `g`.
    generation: u64,
    /// The generation of the last eigendecomposition.
    decomposed_at: u64,
    /// `σ_0 · max(diag D_0)`, the reference of TolXUp.
    initial_scale: f64,
    /// The TolX threshold.
    tol_x: f64,
    /// Whether the last decomposition found a non-positive eigenvalue.
    degenerate: bool,
    /// The standard normal samples `z_k` of the current generation.
    z: Vec<Vec<f64>>,
    /// The steps `y_k = B D z_k` of the current generation.
    y: Vec<Vec<f64>>,
    /// The candidates `x_k = m + σ y_k` of the current generation.
    x: Vec<Vec<f64>>,
    /// Whether `ask` was called since the last `tell`.
    asked: bool,
    /// Cost history for the termination criteria.
    history: History,
}

impl CmaEs {
    /// Starts a run at `mean` with step size `sigma` and `C = I`.
    ///
    /// The tutorial advises that the optimum should lie within `mean ± 3σ`; for a search
    /// interval `[a, b]` per coordinate, a uniform mean and `σ = 0.3 (b - a)` work well.
    ///
    /// # Panics
    ///
    /// Panics if `mean` is empty, `sigma` is not positive and finite, or the configured λ is
    /// below 2.
    #[must_use]
    pub fn new(mean: Vec<f64>, sigma: f64, config: Config) -> Self {
        let n = mean.len();

        Self::with_covariance(mean, sigma, eigen::identity(n), config)
    }

    /// Starts a run with a given covariance matrix (row-major `n × n`, symmetric positive
    /// definite), for example to warm-start from a previous run's [`CmaEs::covariance`] or
    /// to encode different search ranges per coordinate (`c_ii = Δs_i²`).
    ///
    /// # Panics
    ///
    /// Panics as [`CmaEs::new`], or if `covariance` has the wrong size.
    #[must_use]
    pub fn with_covariance(
        mean: Vec<f64>,
        sigma: f64,
        covariance: Vec<f64>,
        config: Config,
    ) -> Self {
        let n = mean.len();
        assert!(n > 0, "CMA-ES needs at least one dimension");
        assert!(
            sigma > 0.0 && sigma.is_finite(),
            "the step size must be positive and finite"
        );
        assert_eq!(covariance.len(), n * n, "the covariance must be n × n");

        let lambda = config
            .lambda
            .unwrap_or_else(|| Parameters::default_lambda(n));
        let params = Parameters::new(n, lambda);
        let tol_x = config.tol_x.unwrap_or(1e-12 * sigma);
        let mut cma = Self {
            config,
            params,
            mean,
            sigma,
            covariance,
            basis: Vec::new(),
            scales: Vec::new(),
            path_sigma: alloc::vec![0.0; n],
            path_c: alloc::vec![0.0; n],
            generation: 0,
            decomposed_at: 0,
            initial_scale: 0.0,
            tol_x,
            degenerate: false,
            z: Vec::new(),
            y: Vec::new(),
            x: Vec::new(),
            asked: false,
            history: History::default(),
        };
        cma.decompose();
        cma.initial_scale = cma.sigma * cma.max_scale();

        cma
    }

    /// The strategy parameters in use.
    #[must_use]
    pub fn parameters(&self) -> &Parameters {
        &self.params
    }

    /// The distribution mean `m`, the current estimate of the optimum.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The step size `σ`.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// The covariance matrix `C`, row-major `n × n`.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Completed generations.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The condition number of `C` from the last decomposition, `+∞` if degenerate.
    #[must_use]
    pub fn condition(&self) -> f64 {
        if self.degenerate {
            return f64::INFINITY;
        }

        let max = self.max_scale();
        let min = self.scales.iter().copied().fold(f64::INFINITY, f64::min);

        (max / min) * (max / min)
    }

    /// Moves the mean, keeping σ, C and the evolution paths: a warm start for a shifted
    /// problem, such as the next control period of a receding-horizon planner.
    ///
    /// # Panics
    ///
    /// Panics if `mean` has the wrong dimension.
    pub fn set_mean(&mut self, mean: Vec<f64>) {
        assert_eq!(
            mean.len(),
            self.mean.len(),
            "the mean must keep its dimension"
        );
        self.mean = mean;
        self.asked = false;
    }

    /// Samples the λ candidates of the next generation, (38)–(40).
    pub fn ask<R: Rng + ?Sized>(&mut self, rng: &mut R) -> &[Vec<f64>] {
        let n = self.params.n;
        let lambda = self.params.lambda;
        self.z.resize_with(lambda, || alloc::vec![0.0; n]);
        self.y.resize_with(lambda, || alloc::vec![0.0; n]);
        self.x.resize_with(lambda, || alloc::vec![0.0; n]);
        for k in 0..lambda {
            for zi in &mut self.z[k] {
                *zi = standard_normal(rng);
            }

            for i in 0..n {
                let mut yi = 0.0;
                for j in 0..n {
                    yi += self.basis[i * n + j] * self.scales[j] * self.z[k][j];
                }
                self.y[k][i] = yi;
                self.x[k][i] = self.mean[i] + self.sigma * yi;
            }
        }

        self.asked = true;
        &self.x
    }

    /// Updates the distribution from the costs of the last [`CmaEs::ask`]ed candidates, in
    /// the same order, (41)–(47); returns the termination criterion that fired, if any.
    ///
    /// NaN costs count as `+∞`.
    ///
    /// # Panics
    ///
    /// Panics if `ask` was not called since the last `tell`, or `costs.len() != λ`.
    pub fn tell(&mut self, costs: &[f64]) -> Option<Termination> {
        assert!(self.asked, "tell without ask");
        assert_eq!(costs.len(), self.params.lambda, "one cost per candidate");
        self.asked = false;

        let costs: Vec<f64> = costs.iter().map(|&c| sanitize(c)).collect();
        let order = ranking(&costs);
        self.update(&order);
        self.generation += 1;
        if self.generation - self.decomposed_at >= self.params.eigen_interval() {
            self.decompose();
        }

        let sorted: Vec<f64> = order.iter().map(|&k| costs[k]).collect();
        self.history.push(&sorted);

        self.termination(&sorted)
    }

    /// The distribution update (41)–(47) for the ranking `order` of the current candidates.
    fn update(&mut self, order: &[usize]) {
        let p = &self.params;
        let n = p.n;
        let mut y_w = alloc::vec![0.0; n];
        let mut z_w = alloc::vec![0.0; n];
        for (i, &k) in order.iter().take(p.mu).enumerate() {
            for j in 0..n {
                y_w[j] += p.weights[i] * self.y[k][j];
                z_w[j] += p.weights[i] * self.z[k][j];
            }
        }

        for j in 0..n {
            self.mean[j] += p.c_m * self.sigma * y_w[j];
        }

        let norm_ps = self.update_path_sigma(&z_w);
        let h_sigma = self.h_sigma(norm_ps);
        let p = &self.params;
        let factor = h_sigma * libm::sqrt(p.c_c * (2.0 - p.c_c) * p.mu_eff);
        for j in 0..n {
            self.path_c[j] = (1.0 - p.c_c) * self.path_c[j] + factor * y_w[j];
        }

        self.update_covariance(order, h_sigma);
        let p = &self.params;
        self.sigma *= libm::exp(p.c_sigma / p.d_sigma * (norm_ps / p.chi_n - 1.0));
    }

    /// The step-size path update (43), with `C^{-1/2} ⟨y⟩_w = B ⟨z⟩_w`; returns `‖p_σ‖`.
    fn update_path_sigma(&mut self, z_w: &[f64]) -> f64 {
        let p = &self.params;
        let n = p.n;
        let factor = libm::sqrt(p.c_sigma * (2.0 - p.c_sigma) * p.mu_eff);
        let mut norm = 0.0;
        for i in 0..n {
            let bz: f64 = (0..n).map(|j| self.basis[i * n + j] * z_w[j]).sum();
            self.path_sigma[i] = (1.0 - p.c_sigma) * self.path_sigma[i] + factor * bz;
            norm += self.path_sigma[i] * self.path_sigma[i];
        }

        libm::sqrt(norm)
    }

    /// The Heaviside function `h_σ`, using the generation number `g + 1` of this update.
    fn h_sigma(&self, norm_ps: f64) -> f64 {
        let p = &self.params;
        #[expect(
            clippy::cast_precision_loss,
            reason = "generation counts are far below 2^52"
        )]
        let g1 = (self.generation + 1) as f64;
        let correction = libm::sqrt(1.0 - libm::pow(1.0 - p.c_sigma, 2.0 * g1));
        #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
        let threshold = (1.4 + 2.0 / (p.n as f64 + 1.0)) * p.chi_n;

        if norm_ps / correction < threshold {
            1.0
        } else {
            0.0
        }
    }

    /// The covariance update (46)–(47).
    fn update_covariance(&mut self, order: &[usize], h_sigma: f64) {
        let p = &self.params;
        let n = p.n;
        #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
        let nf = n as f64;
        let delta = (1.0 - h_sigma) * p.c_c * (2.0 - p.c_c);
        let decay = 1.0 + p.c_1 * delta - p.c_1 - p.c_mu * p.weight_sum();
        let rank_weights: Vec<f64> = order
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let w = p.weights[i];
                if w >= 0.0 {
                    w
                } else {
                    // ‖C^{-1/2} y‖² = ‖B z‖² = ‖z‖², since B is orthonormal.
                    let z2: f64 = self.z[k].iter().map(|v| v * v).sum();
                    w * nf / z2
                }
            })
            .collect();

        for i in 0..n {
            for j in i..n {
                let mut rank_mu = 0.0;
                for (rank, &k) in order.iter().enumerate() {
                    rank_mu += rank_weights[rank] * self.y[k][i] * self.y[k][j];
                }

                let value = decay * self.covariance[i * n + j]
                    + p.c_1 * self.path_c[i] * self.path_c[j]
                    + p.c_mu * rank_mu;
                self.covariance[i * n + j] = value;
                self.covariance[j * n + i] = value;
            }
        }
    }

    /// Recomputes `B` and `D` from `C`.
    fn decompose(&mut self) {
        let n = self.params.n;
        let eigen = eigen::decompose(&self.covariance, n);
        self.degenerate = eigen.values.iter().any(|&v| !(v > 0.0 && v.is_finite()));
        self.scales = eigen
            .values
            .iter()
            .map(|&v| libm::sqrt(v.max(f64::MIN_POSITIVE)))
            .collect();
        self.basis = eigen.vectors;
        self.decomposed_at = self.generation;
    }

    /// The largest diagonal element of `D`.
    fn max_scale(&self) -> f64 {
        self.scales.iter().copied().fold(0.0, f64::max)
    }

    /// Checks the criteria of Appendix B.3 in the tutorial's order.
    fn termination(&self, sorted_costs: &[f64]) -> Option<Termination> {
        let n = self.params.n;
        let window = self.fun_window();
        let checks = [
            (Termination::NoEffectAxis, self.no_effect_axis()),
            (Termination::NoEffectCoord, self.no_effect_coord()),
            (
                Termination::ConditionCov,
                self.condition() > self.config.max_condition,
            ),
            (
                Termination::EqualFunValues,
                self.history.recent_best_range(window) == Some(0.0),
            ),
            (
                Termination::Stagnation,
                self.history.stagnated(120 + window - 10, self.generation),
            ),
            (
                Termination::TolXUp,
                self.sigma * self.max_scale() > self.config.tol_x_up * self.initial_scale,
            ),
            (Termination::TolFun, self.tol_fun(sorted_costs, window)),
            (Termination::TolX, self.tol_x_reached(n)),
        ];

        checks
            .into_iter()
            .find_map(|(criterion, fired)| fired.then_some(criterion))
    }

    /// The window `10 + ⌈30 n / λ⌉` of EqualFunValues and TolFun.
    fn fun_window(&self) -> usize {
        10 + (30 * self.params.n).div_ceil(self.params.lambda)
    }

    /// NoEffectAxis: `m == m + 0.1 σ d_i b_i` for `i = g mod n`.
    #[expect(
        clippy::float_cmp,
        reason = "the criterion is exact floating-point equality"
    )]
    fn no_effect_axis(&self) -> bool {
        let n = self.params.n;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the remainder is below n, a usize"
        )]
        let axis = (self.generation % n as u64) as usize;
        let step = 0.1 * self.sigma * self.scales[axis];

        (0..n).all(|k| self.mean[k] + step * self.basis[k * n + axis] == self.mean[k])
    }

    /// NoEffectCoord: `m_i == m_i + 0.2 σ sqrt(c_ii)` for some `i`.
    #[expect(
        clippy::float_cmp,
        reason = "the criterion is exact floating-point equality"
    )]
    fn no_effect_coord(&self) -> bool {
        let n = self.params.n;

        (0..n).any(|i| {
            let step = 0.2 * self.sigma * libm::sqrt(self.covariance[i * n + i]);
            self.mean[i] + step == self.mean[i]
        })
    }

    /// TolFun: recent best costs and the current costs all within `tol_fun`.
    fn tol_fun(&self, sorted_costs: &[f64], window: usize) -> bool {
        let Some((low, high)) = self.history.recent_best_bounds(window) else {
            return false;
        };

        let low = low.min(sorted_costs[0]);
        let high = high.max(sorted_costs[sorted_costs.len() - 1]);

        high - low < self.config.tol_fun
    }

    /// TolX: every `σ sqrt(c_ii)` and every `|σ p_c,i|` below the threshold.
    fn tol_x_reached(&self, n: usize) -> bool {
        (0..n).all(|i| {
            self.sigma * libm::sqrt(self.covariance[i * n + i]) < self.tol_x
                && (self.sigma * self.path_c[i]).abs() < self.tol_x
        })
    }

    /// Minimises `objective` until the budget or a termination criterion stops the run.
    ///
    /// A generation interrupted by the budget is not told; every evaluated candidate still
    /// counts for [`Outcome::best`]. The population of the outcome is the last complete
    /// generation (repaired points with their objective values).
    pub fn minimize<O, B, Obs, R>(
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
        let mut evaluator = Evaluator::new(objective, budget, observer);
        let (stop, population) = self.run(&mut evaluator, rng);

        evaluator.outcome(population, stop)
    }

    /// The generation loop behind [`CmaEs::minimize`] and [`Ipop`].
    fn run<O, B, Obs, R>(
        &mut self,
        evaluator: &mut Evaluator<'_, Vec<f64>, O, B, Obs>,
        rng: &mut R,
    ) -> (Stop, Population)
    where
        O: Objective<Vec<f64>> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let mut last = Vec::new();
        loop {
            let Some((ranks, population)) = self.evaluate_generation(evaluator, rng) else {
                return (Stop::Budget, last);
            };

            let termination = self.tell(&ranks);
            evaluator.end_generation(crate::ga::population_best(&population));
            last = population;
            if let Some(criterion) = termination {
                return (Stop::CmaEs(criterion), last);
            }
        }
    }

    /// Asks for and evaluates one generation: returns the ranking costs (with the bound
    /// penalty) and the evaluated repaired points, or `None` if the budget ran out.
    fn evaluate_generation<O, B, Obs, R>(
        &mut self,
        evaluator: &mut Evaluator<'_, Vec<f64>, O, B, Obs>,
        rng: &mut R,
    ) -> Option<(Vec<f64>, Population)>
    where
        O: Objective<Vec<f64>> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let candidates = self.ask(rng).to_vec();
        let mut ranks = Vec::with_capacity(candidates.len());
        let mut population = Vec::with_capacity(candidates.len());
        for x in candidates {
            let (repaired, penalty) = match &self.config.bounds {
                Some(bounds) => {
                    let mut repaired = x.clone();
                    bounds.clamp(&mut repaired);
                    let distance: f64 = x
                        .iter()
                        .zip(&repaired)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (repaired, self.config.penalty * distance)
                }
                None => (x, 0.0),
            };

            let cost = evaluator.evaluate(&repaired)?;
            ranks.push(cost + penalty);
            population.push(Individual {
                genome: repaired,
                cost,
            });
        }

        Some((ranks, population))
    }
}

/// IPOP-CMA-ES (Auger and Hansen 2005): independent restarts with the population size
/// multiplied by `factor` each time, from a uniform mean in `bounds` with `σ = sigma_fraction
/// · (b - a)` (mean width), until the budget is exhausted or `max_restarts` is reached.
///
/// ```
/// use evolutionary_planning::cmaes::{Config, Ipop};
/// use evolutionary_planning::operators::real::Bounds;
/// use evolutionary_planning::problems::single::rastrigin;
/// use evolutionary_planning::{Limits, SplitMix64};
///
/// let mut ipop = Ipop::new(Config::default(), Bounds::uniform(3, -5.0, 5.0));
/// let mut objective = |x: &Vec<f64>| rastrigin(x);
/// let outcome = ipop.run(&mut objective, &mut Limits::evaluations(100_000).with_target(1e-8), &mut (), &mut SplitMix64::new(3));
/// assert!(outcome.best_cost() <= 1e-8);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Ipop {
    /// The options of every restart; its `lambda` is the first population size.
    pub config: Config,
    /// The box the initial means are drawn from.
    pub bounds: Bounds,
    /// The initial σ as a fraction of the mean bound width. Default `0.3`.
    pub sigma_fraction: f64,
    /// The population growth per restart. Default `2`.
    pub factor: f64,
    /// The largest number of restarts after the first run. Default `9`.
    pub max_restarts: u64,
}

impl Ipop {
    /// IPOP with the defaults: `σ = 0.3 (b - a)`, doubling λ, at most nine restarts.
    #[must_use]
    pub fn new(config: Config, bounds: Bounds) -> Self {
        Self {
            config,
            bounds,
            sigma_fraction: 0.3,
            factor: 2.0,
            max_restarts: 9,
        }
    }

    /// Runs restarts until the budget stops them or `max_restarts` restarts have terminated.
    ///
    /// # Panics
    ///
    /// Panics if the bounds have dimension `0` or zero width.
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
        let n = self.bounds.dimension();
        #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
        let width = (0..n).map(|i| self.bounds.width(i)).sum::<f64>() / n as f64;
        let sigma = self.sigma_fraction * width;
        let mut lambda = self
            .config
            .lambda
            .unwrap_or_else(|| Parameters::default_lambda(n));
        let mut evaluator = Evaluator::new(objective, budget, observer);
        let mut restart = 0;
        loop {
            let mean = (0..n)
                .map(|i| uniform(rng, self.bounds.lower()[i], self.bounds.upper()[i]))
                .collect();
            let config = Config {
                lambda: Some(lambda),
                ..self.config.clone()
            };
            let mut cma = CmaEs::new(mean, sigma, config);
            let (stop, population) = cma.run(&mut evaluator, rng);
            if stop == Stop::Budget || restart >= self.max_restarts {
                return evaluator.outcome(population, stop);
            }

            restart += 1;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss,
                reason = "population sizes are small positive integers"
            )]
            {
                lambda = libm::round(lambda as f64 * self.factor) as usize;
            }
            evaluator.emit(Event::Restart {
                restart,
                population: lambda,
            });
        }
    }
}

#[cfg(test)]
mod tests;
