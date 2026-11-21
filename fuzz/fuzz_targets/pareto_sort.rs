//! Fuzzes [`non_dominated_sort`] and [`crowding_distance`] on arbitrary floats, infinities
//! included: the fronts partition the points, no point is dominated within or after its own
//! front, every later point is dominated by its predecessor front, and crowding distances
//! are never NaN or negative.

#![no_main]
#![forbid(unsafe_code)]

use evolutionary_planning::pareto::{constrained_dominates, crowding_distance, non_dominated_sort};
use libfuzzer_sys::fuzz_target;

/// Most points per input; the sort is quadratic.
const MAX_POINTS: usize = 64;

/// NaN as `+∞`, the mapping the algorithms apply before sorting: with NaN the dominance
/// relation can be cyclic, which `non_dominated_sort` does not support.
fn sanitize(x: f64) -> f64 {
    if x.is_nan() { f64::INFINITY } else { x }
}

/// The front index of every point; panics if a point is missing or appears twice.
fn ranks(fronts: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut rank = vec![usize::MAX; n];
    for (k, front) in fronts.iter().enumerate() {
        for &i in front {
            assert_eq!(rank[i], usize::MAX, "point {i} in two fronts");
            rank[i] = k;
        }
    }

    assert!(
        rank.iter().all(|&r| r != usize::MAX),
        "a point is in no front"
    );

    rank
}

/// Checks that `i` is not dominated by any point of its own or a later front and, past the
/// first front, is dominated by some point of the front before.
fn check_rank(points: &[[f64; 3]], violations: &[f64], rank: &[usize], i: usize) {
    let dominates =
        |j: usize| constrained_dominates(&points[j], violations[j], &points[i], violations[i]);

    assert!(
        !(0..points.len()).any(|j| rank[j] >= rank[i] && dominates(j)),
        "point {i} dominated within or after its front"
    );
    assert!(
        rank[i] == 0 || (0..points.len()).any(|j| rank[j] + 1 == rank[i] && dominates(j)),
        "point {i} not dominated by the previous front"
    );
}

fuzz_target!(|input: Vec<([f64; 3], f64)>| {
    let input = &input[..input.len().min(MAX_POINTS)];
    let points: Vec<[f64; 3]> = input.iter().map(|(p, _)| p.map(sanitize)).collect();
    let violations: Vec<f64> = input.iter().map(|&(_, v)| sanitize(v)).collect();

    let fronts = non_dominated_sort(&points, &violations);
    let rank = ranks(&fronts, points.len());
    for i in 0..points.len() {
        check_rank(&points, &violations, &rank, i);
    }

    for front in &fronts {
        assert!(front.is_sorted(), "front not ascending: {front:?}");
        let distances = crowding_distance(&points, front);
        assert_eq!(distances.len(), front.len());
        assert!(
            distances.iter().all(|d| *d >= 0.0),
            "NaN or negative crowding distance: {distances:?}"
        );
    }
});
