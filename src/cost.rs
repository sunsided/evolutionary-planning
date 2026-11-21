//! Costs and evaluated individuals: the NaN policy, the total order used for every sort, and
//! the [`Individual`] pair of a genome and its cost.
//!
//! Every objective in this crate is minimised (see `docs/design.md`).

use alloc::vec::Vec;
use core::cmp::Ordering;

/// Maps NaN to `+∞`, so a broken objective yields the worst possible individual instead of
/// poisoning comparisons. Applied once to every value an objective returns.
///
/// ```
/// use evolutionary_planning::sanitize;
///
/// assert_eq!(sanitize(f64::NAN), f64::INFINITY);
/// assert_eq!(sanitize(-1.5), -1.5);
/// ```
#[must_use]
pub fn sanitize(cost: f64) -> f64 {
    if cost.is_nan() { f64::INFINITY } else { cost }
}

/// A genome together with its (sanitised) cost; smaller is better.
#[derive(Debug, Clone, PartialEq)]
pub struct Individual<G> {
    /// The candidate solution.
    pub genome: G,
    /// Its cost as returned by the objective, NaN mapped to `+∞`.
    pub cost: f64,
}

impl<G> Individual<G> {
    /// Pairs `genome` with `cost`, sanitising the cost.
    #[must_use]
    pub fn new(genome: G, cost: f64) -> Self {
        Self {
            genome,
            cost: sanitize(cost),
        }
    }
}

/// Compares two costs; costs are sanitised, so `total_cmp` orders them like `<` would.
pub(crate) fn compare(a: f64, b: f64) -> Ordering {
    a.total_cmp(&b)
}

/// Returns the indices of `costs` sorted ascending by cost, ties broken by index, so the
/// order never depends on the sort algorithm.
#[must_use]
pub fn ranking(costs: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..costs.len()).collect();
    order.sort_by(|&a, &b| compare(costs[a], costs[b]).then(a.cmp(&b)));

    order
}

/// Sorts `population` ascending by cost; stable, so ties keep their order.
pub(crate) fn sort_population<G>(population: &mut [Individual<G>]) {
    population.sort_by(|a, b| compare(a.cost, b.cost));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ties are broken by index and NaN sorts last after sanitising.
    #[test]
    fn ranking_is_stable_and_nan_last() {
        let costs = [2.0, 1.0, sanitize(f64::NAN), 1.0, -0.0];

        assert_eq!(ranking(&costs), [4, 1, 3, 0, 2]);
    }
}
