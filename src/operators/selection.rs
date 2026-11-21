//! Parent selection over a slice of costs (smaller is better): tournament, truncation,
//! fitness-proportional, linear ranking, stochastic universal sampling and uniform.

use alloc::vec::Vec;

use rand_core::Rng;

use crate::cost::{compare, ranking};
use crate::rng::{below, unit_f64};

/// Picks parent indices from a population, given the population's costs.
pub trait Selection {
    /// Appends `count` indices into `costs` to `out`. Indices may repeat.
    ///
    /// `costs` is never empty when the algorithms call this.
    fn select<R: Rng + ?Sized>(
        &mut self,
        costs: &[f64],
        count: usize,
        rng: &mut R,
        out: &mut Vec<usize>,
    );
}

impl<S: Selection + ?Sized> Selection for &mut S {
    fn select<R: Rng + ?Sized>(
        &mut self,
        costs: &[f64],
        count: usize,
        rng: &mut R,
        out: &mut Vec<usize>,
    ) {
        (**self).select(costs, count, rng, out);
    }
}

/// Deterministic `k`-tournament: draw `size` contestants uniformly with replacement, keep
/// the cheapest (the earliest drawn among equals).
///
/// ```
/// use evolutionary_planning::operators::{Selection, selection::Tournament};
/// use evolutionary_planning::SplitMix64;
///
/// let mut picked = Vec::new();
/// Tournament { size: 2 }.select(&[3.0, 1.0, 2.0], 5, &mut SplitMix64::new(1), &mut picked);
/// assert_eq!(picked.len(), 5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tournament {
    /// The number of contestants, at least `1`; `2` is the usual binary tournament.
    pub size: usize,
}

impl Default for Tournament {
    fn default() -> Self {
        Self { size: 2 }
    }
}

impl Selection for Tournament {
    fn select<R: Rng + ?Sized>(
        &mut self,
        costs: &[f64],
        count: usize,
        rng: &mut R,
        out: &mut Vec<usize>,
    ) {
        for _ in 0..count {
            let mut winner = below(rng, costs.len());
            for _ in 1..self.size.max(1) {
                let contestant = below(rng, costs.len());
                if compare(costs[contestant], costs[winner]).is_lt() {
                    winner = contestant;
                }
            }

            out.push(winner);
        }
    }
}

/// Truncation selection: uniform among the best `fraction` of the population (at least one).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Truncation {
    /// The fraction of the population eligible, in `(0, 1]`.
    pub fraction: f64,
}

impl Default for Truncation {
    fn default() -> Self {
        Self { fraction: 0.5 }
    }
}

impl Selection for Truncation {
    fn select<R: Rng + ?Sized>(
        &mut self,
        costs: &[f64],
        count: usize,
        rng: &mut R,
        out: &mut Vec<usize>,
    ) {
        let order = ranking(costs);
        let eligible = fraction_count(self.fraction, costs.len());
        for _ in 0..count {
            out.push(order[below(rng, eligible)]);
        }
    }
}

/// Returns `ceil(fraction * n)` clamped to `1..=n`.
fn fraction_count(fraction: f64, n: usize) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a population fraction, clamped to 1..=n"
    )]
    let count = libm::ceil(fraction.clamp(0.0, 1.0) * n as f64) as usize;

    count.clamp(1, n)
}

/// How a selection scheme turns weights into indices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Sampling {
    /// Independent roulette-wheel spins, one per pick.
    #[default]
    Roulette,
    /// Stochastic universal sampling (Baker 1987): one spin with `count` equally spaced
    /// pointers, which has minimal spread around the expected number of copies.
    Universal,
}

/// Fitness-proportional selection on minimised costs: the weight of individual `i` is
/// `worst - cost_i`, so the worst individual is never chosen unless all are equal (then the
/// choice is uniform). Infinite costs get weight zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Proportional {
    /// Roulette or stochastic universal sampling.
    pub sampling: Sampling,
}

impl Selection for Proportional {
    fn select<R: Rng + ?Sized>(
        &mut self,
        costs: &[f64],
        count: usize,
        rng: &mut R,
        out: &mut Vec<usize>,
    ) {
        let worst = costs
            .iter()
            .copied()
            .filter(|c| c.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = costs
            .iter()
            .map(|&c| if c.is_finite() { worst - c } else { 0.0 })
            .collect();

        sample(&weights, count, self.sampling, rng, out);
    }
}

/// Linear ranking selection (Baker 1985) with selective pressure `s ∈ [1, 2]`: the best of
/// `n` individuals is expected to be chosen `s` times per `n` picks, the worst `2 - s`
/// times. The Evolutionary Programmer's scores 1.2 (best) to 0.8 (worst) are `s = 1.2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearRanking {
    /// The selective pressure, clamped to `[1, 2]`.
    pub pressure: f64,
    /// Roulette or stochastic universal sampling.
    pub sampling: Sampling,
}

impl Default for LinearRanking {
    fn default() -> Self {
        Self {
            pressure: 1.5,
            sampling: Sampling::Roulette,
        }
    }
}

impl Selection for LinearRanking {
    fn select<R: Rng + ?Sized>(
        &mut self,
        costs: &[f64],
        count: usize,
        rng: &mut R,
        out: &mut Vec<usize>,
    ) {
        let weights = ranking_weights(costs, self.pressure);

        sample(&weights, count, self.sampling, rng, out);
    }
}

/// Returns the linear ranking weights (expected copies per pick) of every index.
pub(crate) fn ranking_weights(costs: &[f64], pressure: f64) -> Vec<f64> {
    let n = costs.len();
    let s = pressure.clamp(1.0, 2.0);
    let mut weights = alloc::vec![0.0; n];
    if n == 1 {
        weights[0] = 1.0;
        return weights;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "population sizes are far below 2^52"
    )]
    let last = (n - 1) as f64;
    for (rank, &index) in ranking(costs).iter().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "population sizes are far below 2^52"
        )]
        let position = (last - rank as f64) / last;
        weights[index] = 2.0 - s + 2.0 * (s - 1.0) * position;
    }

    weights
}

/// Uniform selection, ignoring costs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Uniform;

impl Selection for Uniform {
    fn select<R: Rng + ?Sized>(
        &mut self,
        costs: &[f64],
        count: usize,
        rng: &mut R,
        out: &mut Vec<usize>,
    ) {
        for _ in 0..count {
            out.push(below(rng, costs.len()));
        }
    }
}

/// Draws `count` indices with probability proportional to `weights` (non-negative). If every
/// weight is zero the draw is uniform.
pub(crate) fn sample<R: Rng + ?Sized>(
    weights: &[f64],
    count: usize,
    sampling: Sampling,
    rng: &mut R,
    out: &mut Vec<usize>,
) {
    let total: f64 = weights.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        for _ in 0..count {
            out.push(below(rng, weights.len()));
        }
        return;
    }

    match sampling {
        Sampling::Roulette => {
            for _ in 0..count {
                out.push(spin(weights, total * unit_f64(rng)));
            }
        }
        Sampling::Universal => universal(weights, total, count, rng, out),
    }
}

/// Returns the index whose cumulative weight interval contains `point`.
fn spin(weights: &[f64], point: f64) -> usize {
    let mut cumulative = 0.0;
    let mut last_positive = 0;
    for (index, &weight) in weights.iter().enumerate() {
        if weight > 0.0 {
            last_positive = index;
        }

        cumulative += weight;
        if point < cumulative {
            return index;
        }
    }

    // Rounding left `point` past the last interval.
    last_positive
}

/// Stochastic universal sampling: `count` pointers spaced `total / count` apart.
fn universal<R: Rng + ?Sized>(
    weights: &[f64],
    total: f64,
    count: usize,
    rng: &mut R,
    out: &mut Vec<usize>,
) {
    if count == 0 {
        return;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "selection counts are far below 2^52"
    )]
    let spacing = total / count as f64;
    let mut pointer = spacing * unit_f64(rng);
    let mut cumulative = weights[0];
    let mut index = 0;
    for _ in 0..count {
        while pointer >= cumulative && index + 1 < weights.len() {
            index += 1;
            cumulative += weights[index];
        }

        out.push(index);
        pointer += spacing;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMix64;

    /// A tournament of the whole population size picks the best almost always.
    #[test]
    fn big_tournament_prefers_best() {
        let costs = [5.0, 1.0, 3.0, 4.0];
        let mut out = Vec::new();
        Tournament { size: 50 }.select(&costs, 100, &mut SplitMix64::new(3), &mut out);

        assert!(out.iter().all(|&i| i == 1));
    }

    /// Truncation picks only from the top fraction.
    #[test]
    fn truncation_stays_in_top() {
        let costs = [5.0, 1.0, 3.0, 4.0];
        let mut out = Vec::new();
        Truncation { fraction: 0.5 }.select(&costs, 100, &mut SplitMix64::new(3), &mut out);

        assert!(out.iter().all(|&i| i == 1 || i == 2));
    }

    /// Ranking weights sum to `n` and span `2 - s ..= s`.
    #[test]
    fn ranking_weights_are_linear() {
        let w = ranking_weights(&[3.0, 1.0, 2.0], 1.2);

        assert!((w.iter().sum::<f64>() - 3.0).abs() < 1e-12);
        assert!((w[1] - 1.2).abs() < 1e-12 && (w[0] - 0.8).abs() < 1e-12);
    }

    /// Universal sampling gives each index its expected count up to one.
    #[test]
    fn universal_has_minimal_spread() {
        let weights = [1.0, 2.0, 3.0, 4.0];
        let mut out = Vec::new();
        sample(
            &weights,
            100,
            Sampling::Universal,
            &mut SplitMix64::new(9),
            &mut out,
        );
        for (index, expected) in [10, 20, 30, 40].into_iter().enumerate() {
            let got = out.iter().filter(|&&i| i == index).count();
            assert!(got.abs_diff(expected) <= 1, "index {index}: {got}");
        }
    }

    /// Proportional selection never picks the unique worst.
    #[test]
    fn proportional_skips_worst() {
        let mut out = Vec::new();
        Proportional::default().select(
            &[1.0, 9.0, 2.0, f64::INFINITY],
            200,
            &mut SplitMix64::new(4),
            &mut out,
        );

        assert!(out.iter().all(|&i| i == 0 || i == 2));
    }
}
