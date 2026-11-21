//! Pareto tools shared by the multi-objective algorithms: dominance and constrained
//! dominance, fast non-dominated sorting, crowding distance, bounded Pareto archives and exact
//! hypervolume. All objectives are minimised.
//!
//! ```
//! use evolutionary_planning::pareto::{dominates, non_dominated_sort};
//!
//! let points = [vec![1.0, 4.0], vec![2.0, 2.0], vec![3.0, 3.0], vec![4.0, 1.0]];
//! assert!(dominates(&points[1], &points[2]));
//!
//! let fronts = non_dominated_sort(&points, &[0.0; 4]);
//! assert_eq!(fronts, vec![vec![0, 1, 3], vec![2]]);
//! ```

mod archive;
mod hypervolume;
mod reference;

use alloc::vec::Vec;
use core::cmp::Ordering;

pub use archive::ParetoArchive;
pub use hypervolume::hypervolume;
pub use reference::{das_dennis, two_layer};

/// A genome with its objective vector and constraint violation (`0` if feasible).
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluated<G> {
    /// The candidate solution.
    pub genome: G,
    /// Its objective values, all minimised; NaN mapped to `+∞`.
    pub objectives: Vec<f64>,
    /// Its constraint violation, `0` when feasible; NaN mapped to `+∞`.
    pub violation: f64,
}

impl<G> Evaluated<G> {
    /// Whether the genome satisfies all constraints.
    #[must_use]
    pub fn is_feasible(&self) -> bool {
        self.violation <= 0.0
    }
}

/// Returns `true` if `a` Pareto-dominates `b`: no worse in every objective and strictly
/// better in at least one.
#[must_use]
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly = false;
    for (x, y) in a.iter().zip(b) {
        if x > y {
            return false;
        }

        strictly |= x < y;
    }

    strictly
}

/// Constrained domination (Deb 2000): a feasible point dominates an infeasible one; of two
/// infeasible points the one with smaller violation dominates; of two feasible points Pareto
/// dominance decides.
#[must_use]
pub fn constrained_dominates(a: &[f64], violation_a: f64, b: &[f64], violation_b: f64) -> bool {
    match (violation_a <= 0.0, violation_b <= 0.0) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => violation_a < violation_b,
        (true, true) => dominates(a, b),
    }
}

/// Fast non-dominated sorting (Deb, Pratap, Agarwal and Meyarivan 2002) under constrained
/// domination: returns the fronts as index lists, best front first, each sorted ascending.
/// Equal points share a front.
///
/// `O(m n²)` for `n` points with `m` objectives.
///
/// # Panics
///
/// Panics if `violations` is shorter than `points`.
#[must_use]
pub fn non_dominated_sort<P: AsRef<[f64]>>(points: &[P], violations: &[f64]) -> Vec<Vec<usize>> {
    let n = points.len();
    let mut dominated_by: Vec<Vec<usize>> = alloc::vec![Vec::new(); n];
    let mut counts = alloc::vec![0usize; n];
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (points[i].as_ref(), points[j].as_ref());
            if constrained_dominates(a, violations[i], b, violations[j]) {
                dominated_by[i].push(j);
                counts[j] += 1;
            } else if constrained_dominates(b, violations[j], a, violations[i]) {
                dominated_by[j].push(i);
                counts[i] += 1;
            }
        }
    }

    let mut fronts = Vec::new();
    let mut current: Vec<usize> = (0..n).filter(|&i| counts[i] == 0).collect();
    while !current.is_empty() {
        let mut next = Vec::new();
        for &i in &current {
            for &j in &dominated_by[i] {
                counts[j] -= 1;
                if counts[j] == 0 {
                    next.push(j);
                }
            }
        }

        next.sort_unstable();
        fronts.push(current);
        current = next;
    }

    fronts
}

/// The indices (ascending) of the points that no other point dominates, by an `O(n²)` scan;
/// the brute-force reference for tests. Constraints are ignored.
#[must_use]
pub fn non_dominated<P: AsRef<[f64]>>(points: &[P]) -> Vec<usize> {
    (0..points.len())
        .filter(|&i| {
            !points
                .iter()
                .any(|q| dominates(q.as_ref(), points[i].as_ref()))
        })
        .collect()
}

/// The crowding distance of each member of `front` (indices into `points`), in the order of
/// `front`: the normalised perimeter of the cuboid spanned by its neighbours per objective;
/// boundary points get `+∞`. Objectives with zero range contribute nothing.
#[must_use]
pub fn crowding_distance<P: AsRef<[f64]>>(points: &[P], front: &[usize]) -> Vec<f64> {
    let k = front.len();
    let mut distance = alloc::vec![0.0; k];
    if k <= 2 {
        distance.fill(f64::INFINITY);
        return distance;
    }

    let m = points[front[0]].as_ref().len();
    let mut order: Vec<usize> = (0..k).collect();
    for objective in 0..m {
        let value = |slot: usize| points[front[slot]].as_ref()[objective];
        order.sort_by(|&a, &b| value(a).total_cmp(&value(b)).then(a.cmp(&b)));
        let range = value(order[k - 1]) - value(order[0]);
        distance[order[0]] = f64::INFINITY;
        distance[order[k - 1]] = f64::INFINITY;
        if !(range > 0.0 && range.is_finite()) {
            continue;
        }

        for w in 1..k - 1 {
            distance[order[w]] += (value(order[w + 1]) - value(order[w - 1])) / range;
        }
    }

    distance
}

/// Orders by rank (smaller first), then crowding distance (larger first), then index: the
/// crowded-comparison operator of NSGA-II.
pub(crate) fn crowded_order(rank_a: usize, crowd_a: f64, rank_b: usize, crowd_b: f64) -> Ordering {
    rank_a.cmp(&rank_b).then(crowd_b.total_cmp(&crowd_a))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// Fast non-dominated sorting agrees with repeatedly peeling the brute-force
        /// non-dominated set, including duplicates and constraint violations.
        #[test]
        fn sort_matches_peeling(
            raw in proptest::collection::vec((0u8..5, 0u8..5, 0u8..3), 0..40),
        ) {
            let points: Vec<Vec<f64>> = raw.iter().map(|&(a, b, _)| alloc::vec![f64::from(a), f64::from(b)]).collect();
            let violations: Vec<f64> = raw.iter().map(|&(_, _, v)| if v == 0 { 0.0 } else { f64::from(v) }).collect();
            let fronts = non_dominated_sort(&points, &violations);

            let mut remaining: Vec<usize> = (0..points.len()).collect();
            for front in &fronts {
                let expected: Vec<usize> = remaining
                    .iter()
                    .copied()
                    .filter(|&i| !remaining.iter().any(|&j| constrained_dominates(&points[j], violations[j], &points[i], violations[i])))
                    .collect();
                prop_assert_eq!(front, &expected);
                remaining.retain(|i| !front.contains(i));
            }
            prop_assert!(remaining.is_empty());
        }
    }

    /// Crowding distance of an evenly spaced front.
    #[test]
    fn crowding_of_line() {
        let points: Vec<Vec<f64>> = (0..5)
            .map(|i| alloc::vec![f64::from(i), f64::from(4 - i)])
            .collect();
        let d = crowding_distance(&points, &[0, 1, 2, 3, 4]);

        assert!(d[0].is_infinite() && d[4].is_infinite());
        assert!((d[1] - 1.0).abs() < 1e-12 && (d[2] - 1.0).abs() < 1e-12);
    }

    /// Constrained domination prefers feasibility, then smaller violation.
    #[test]
    fn constraint_domination() {
        assert!(constrained_dominates(&[9.0], 0.0, &[0.0], 1.0));
        assert!(constrained_dominates(&[9.0], 0.5, &[0.0], 1.0));
        assert!(!constrained_dominates(&[1.0], 0.0, &[1.0], 0.0));
    }
}
