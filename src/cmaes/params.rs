//! The strategy parameters of CMA-ES: Table 1 of Hansen's tutorial (the 2016 defaults with
//! negative recombination weights).

use alloc::vec::Vec;

/// The derived strategy parameters for dimension `n` and population size `λ`, computed
/// exactly as in Table 1 of Hansen (2016), equations (48)–(58).
#[derive(Debug, Clone, PartialEq)]
pub struct Parameters {
    /// The dimension `n`.
    pub n: usize,
    /// The population size `λ` (48): `4 + ⌊3 ln n⌋` by default.
    pub lambda: usize,
    /// The number of positive weights `μ = ⌊λ/2⌋`.
    pub mu: usize,
    /// The recombination weights `w_1 ≥ … ≥ w_λ` (53): the first `μ` are positive and sum to
    /// one, the rest are non-positive ("active" CMA).
    pub weights: Vec<f64>,
    /// The variance effective selection mass `μ_eff = (Σ w_i)² / Σ w_i²` over the positive
    /// weights.
    pub mu_eff: f64,
    /// The variance effective selection mass of the negative weights, `μ_eff^-`.
    pub mu_eff_negative: f64,
    /// The learning rate of the mean, `c_m = 1` (54).
    pub c_m: f64,
    /// The cumulation constant of the step-size path, `c_σ` (55).
    pub c_sigma: f64,
    /// The step-size damping `d_σ` (55).
    pub d_sigma: f64,
    /// The cumulation constant of the rank-one path, `c_c` (56).
    pub c_c: f64,
    /// The rank-one learning rate `c_1` (57).
    pub c_1: f64,
    /// The rank-μ learning rate `c_μ` (58).
    pub c_mu: f64,
    /// `E‖N(0, I)‖ ≈ √n (1 - 1/(4n) + 1/(21n²))`.
    pub chi_n: f64,
}

impl Parameters {
    /// The default population size `4 + ⌊3 ln n⌋` (48).
    #[must_use]
    pub fn default_lambda(n: usize) -> usize {
        #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
        let log = libm::log(n.max(1) as f64);

        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a small non-negative count"
        )]
        let extra = libm::floor(3.0 * log) as usize;

        4 + extra
    }

    /// Computes every parameter for dimension `n` and population size `lambda`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` or `lambda < 2`.
    #[must_use]
    pub fn new(n: usize, lambda: usize) -> Self {
        assert!(n > 0, "CMA-ES needs at least one dimension");
        assert!(lambda >= 2, "CMA-ES needs a population of at least two");

        let nf = to_f64(n);
        let mu = lambda / 2;
        let raw = raw_weights(lambda);
        let mu_eff = selection_mass(&raw[..mu]);
        let mu_eff_negative = selection_mass(&raw[mu..]);

        let c_sigma = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (libm::sqrt((mu_eff - 1.0) / (nf + 1.0)) - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
        let alpha_cov = 2.0;
        let c_1 = alpha_cov / ((nf + 1.3) * (nf + 1.3) + mu_eff);
        let c_mu = (1.0 - c_1).min(
            alpha_cov * (0.25 + mu_eff + 1.0 / mu_eff - 2.0)
                / ((nf + 2.0) * (nf + 2.0) + alpha_cov * mu_eff / 2.0),
        );
        let weights = final_weights(&raw, mu, n, c_1, c_mu, mu_eff, mu_eff_negative);
        let chi_n = libm::sqrt(nf) * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        Self {
            n,
            lambda,
            mu,
            weights,
            mu_eff,
            mu_eff_negative,
            c_m: 1.0,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
        }
    }

    /// The sum of all weights `Σ w_j`, which enters the decay factor of (47).
    #[must_use]
    pub fn weight_sum(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// The number of generations between eigendecompositions,
    /// `max(1, ⌊1 / (10 n (c_1 + c_μ))⌋)` (Appendix B.2).
    #[must_use]
    pub fn eigen_interval(&self) -> u64 {
        let interval = libm::floor(1.0 / (10.0 * to_f64(self.n) * (self.c_1 + self.c_mu)));

        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a small non-negative count"
        )]
        let interval = interval as u64;

        interval.max(1)
    }
}

/// Converts a count to `f64`.
fn to_f64(n: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "dimensions and population sizes are far below 2^52"
    )]
    let n = n as f64;

    n
}

/// The preliminary weights `w'_i = ln((λ+1)/2) - ln i` (49).
fn raw_weights(lambda: usize) -> Vec<f64> {
    let base = libm::log(f64::midpoint(to_f64(lambda), 1.0));

    (1..=lambda).map(|i| base - libm::log(to_f64(i))).collect()
}

/// `(Σ w)² / Σ w²`, the variance effective selection mass of a group of weights; `0` for an
/// empty or all-zero group.
fn selection_mass(weights: &[f64]) -> f64 {
    let sum: f64 = weights.iter().sum();
    let squares: f64 = weights.iter().map(|w| w * w).sum();

    if squares > 0.0 {
        sum * sum / squares
    } else {
        0.0
    }
}

/// The final weights (53): positive ones normalised to sum one, negative ones scaled to sum
/// to `-min(α_μ^-, α_μeff^-, α_posdef^-)`.
fn final_weights(
    raw: &[f64],
    mu: usize,
    n: usize,
    c_1: f64,
    c_mu: f64,
    mu_eff: f64,
    mu_eff_negative: f64,
) -> Vec<f64> {
    let positive: f64 = raw.iter().filter(|&&w| w >= 0.0).sum();
    let negative: f64 = raw.iter().filter(|&&w| w < 0.0).map(|w| -w).sum();
    let alpha_mu = 1.0 + c_1 / c_mu;
    let alpha_mu_eff = 1.0 + 2.0 * mu_eff_negative / (mu_eff + 2.0);
    let alpha_pos_def = (1.0 - c_1 - c_mu) / (to_f64(n) * c_mu);
    let scale = alpha_mu.min(alpha_mu_eff).min(alpha_pos_def);
    debug_assert!(
        raw[..mu].iter().all(|&w| w > 0.0),
        "the first mu raw weights are positive"
    );

    raw.iter()
        .map(|&w| {
            if w >= 0.0 {
                w / positive
            } else {
                scale * w / negative
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Figure 7 of the tutorial: default λ and the sum of all weights for n = 5, 10, 25, 100.
    ///
    /// The printed sums (-0.95, -0.63, -0.42, -0.28) differ from Table 1's formulas by up to
    /// 0.04 for small n (Table 1 gives -0.988 and -0.649 for n = 5 and 10, exactly `-c_1 /
    /// c_μ`); the figure was presumably drawn with a slightly different parameter version.
    /// Table 1 is normative here, so the figure is checked to 0.05.
    #[test]
    fn matches_figure_7() {
        for (n, lambda, sum) in [
            (5, 8, -0.95),
            (10, 10, -0.63),
            (25, 13, -0.42),
            (100, 17, -0.28),
        ] {
            let p = Parameters::new(n, Parameters::default_lambda(n));

            assert_eq!(p.lambda, lambda, "n = {n}");
            assert!(
                (p.weight_sum() - sum).abs() < 0.05,
                "n = {n}: {}",
                p.weight_sum()
            );
            assert!((p.weight_sum() + p.c_1 / p.c_mu).abs() < 1e-12, "n = {n}");
        }
    }

    /// Positive weights sum to one; `c_1 + c_μ Σ w_j >= 0` always (C never grows by decay),
    /// with equality when `α_μ^-` is the binding bound, as for the default λ from n = 5 on.
    #[test]
    fn weight_normalisation() {
        for n in [1, 2, 3, 5, 10, 40] {
            for lambda in [Parameters::default_lambda(n), 50] {
                let p = Parameters::new(n, lambda);
                let positive: f64 = p.weights[..p.mu].iter().sum();
                let decay = p.c_1 + p.c_mu * p.weight_sum();

                assert!((positive - 1.0).abs() < 1e-12);
                assert!(decay >= -1e-12, "n = {n}, λ = {lambda}");
                assert!(p.weights.windows(2).all(|w| w[0] >= w[1]));
                if n >= 5 && lambda == Parameters::default_lambda(n) {
                    assert!(decay.abs() < 1e-12, "n = {n}");
                }
            }
        }
    }

    /// The textbook values for n = 10: `μ_eff ≈ 3.17`, and the lazy update interval.
    #[test]
    fn known_values() {
        let p = Parameters::new(10, 10);

        assert_eq!(p.mu, 5);
        assert!((p.mu_eff - 3.1672).abs() < 1e-3, "{}", p.mu_eff);
        assert!((p.chi_n - 3.0847).abs() < 1e-3, "{}", p.chi_n);
        assert!(p.eigen_interval() >= 1);
    }
}
