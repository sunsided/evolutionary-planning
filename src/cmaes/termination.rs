//! The termination criteria of Appendix B.3 of Hansen's tutorial, and the history they need.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// A CMA-ES termination criterion (Hansen 2016, Appendix B.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Termination {
    /// Adding a 0.1-standard deviation vector along a principal axis of C does not change the
    /// mean.
    NoEffectAxis,
    /// Adding 0.2 standard deviations in a coordinate does not change the mean.
    NoEffectCoord,
    /// The condition number of C exceeds [`Config::max_condition`](super::Config).
    ConditionCov,
    /// The best costs of the last `10 + ⌈30 n / λ⌉` generations are all equal.
    EqualFunValues,
    /// Over a long window, the median of the most recent 30 % of both the best and the
    /// median costs is not better than the median of the first 30 %.
    Stagnation,
    /// `σ · max(diag(D))` grew by more than [`Config::tol_x_up`](super::Config): σ was far
    /// too small initially, or the run diverges.
    TolXUp,
    /// The best costs of the last `10 + ⌈30 n / λ⌉` generations and all costs of the current
    /// one lie within a range below [`Config::tol_fun`](super::Config).
    TolFun,
    /// Every coordinate's standard deviation and every component of `σ p_c` is below
    /// [`Config::tol_x`](super::Config).
    TolX,
}

/// The cost history the criteria look at.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct History {
    /// The best cost of each generation, most recent last, at most `MAX_STAGNATION` long.
    pub(crate) best: VecDeque<f64>,
    /// The median cost of each generation, most recent last.
    pub(crate) median: VecDeque<f64>,
}

/// The longest stagnation window of the tutorial: 20 000 generations.
const MAX_STAGNATION: usize = 20_000;

impl History {
    /// Records one generation's sorted costs.
    pub(crate) fn push(&mut self, sorted_costs: &[f64]) {
        self.best.push_back(sorted_costs[0]);
        self.median.push_back(median_of_sorted(sorted_costs));
        if self.best.len() > MAX_STAGNATION {
            self.best.pop_front();
            self.median.pop_front();
        }
    }

    /// The range (max - min) of the last `window` best costs, if that many exist.
    pub(crate) fn recent_best_range(&self, window: usize) -> Option<f64> {
        self.recent_best_bounds(window)
            .map(|(low, high)| high - low)
    }

    /// The smallest and largest of the last `window` best costs, if that many exist.
    pub(crate) fn recent_best_bounds(&self, window: usize) -> Option<(f64, f64)> {
        if self.best.len() < window {
            return None;
        }

        let recent = self.best.iter().skip(self.best.len() - window);

        Some(
            recent.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            }),
        )
    }

    /// The stagnation test over a window of `max(minimum, 20 % of generations)` (capped at
    /// 20 000) generations.
    pub(crate) fn stagnated(&self, minimum: usize, generations: u64) -> bool {
        let fifth = usize::try_from(generations / 5).unwrap_or(usize::MAX);
        let window = minimum.max(fifth).min(MAX_STAGNATION);
        if self.best.len() < window {
            return false;
        }

        stalled(&self.best, window) && stalled(&self.median, window)
    }
}

/// Whether the median of the last 30 % of the window is not better than that of the first
/// 30 %.
fn stalled(values: &VecDeque<f64>, window: usize) -> bool {
    let recent: Vec<f64> = values.iter().skip(values.len() - window).copied().collect();
    let part = (window * 3).div_ceil(10).max(1);

    median(&recent[window - part..]) >= median(&recent[..part])
}

/// The median of an unsorted slice.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    median_of_sorted(&sorted)
}

/// The median of a sorted, non-empty slice.
fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat history stagnates; a steadily improving one does not.
    #[test]
    fn stagnation() {
        let mut flat = History::default();
        let mut improving = History::default();
        for g in 0..200 {
            flat.push(&[1.0, 2.0, 3.0]);
            let v = f64::from(200 - g);
            improving.push(&[v, v + 1.0, v + 2.0]);
        }

        assert!(flat.stagnated(150, 200));
        assert!(!improving.stagnated(150, 200));
        assert_eq!(flat.recent_best_range(50), Some(0.0));
        assert_eq!(flat.recent_best_range(500), None);
    }
}
