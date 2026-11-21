//! The exact hypervolume indicator: the volume dominated by a point set and bounded by a
//! reference point, by a sweep in two dimensions and hypervolume-by-slicing-objectives (HSO,
//! While et al. 2006) in more.

use alloc::vec::Vec;

use super::dominates;

/// The hypervolume of `points` (minimised objectives) with respect to `reference`: the
/// Lebesgue measure of the region dominated by some point and dominating `reference`.
/// Points that do not strictly dominate the reference in every objective contribute nothing.
///
/// Exact; `O(n log n)` in two dimensions, exponential in the dimension otherwise, which is
/// fine for the front sizes of tests and benchmarks.
///
/// ```
/// use evolutionary_planning::pareto::hypervolume;
///
/// let front = [vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
/// assert_eq!(hypervolume(&front, &[4.0, 4.0]), 6.0);
/// ```
#[must_use]
pub fn hypervolume<P: AsRef<[f64]>>(points: &[P], reference: &[f64]) -> f64 {
    let inside: Vec<Vec<f64>> = points
        .iter()
        .map(|p| p.as_ref().to_vec())
        .filter(|p| p.len() == reference.len() && p.iter().zip(reference).all(|(x, r)| x < r))
        .collect();

    slice_volume(inside, reference)
}

/// HSO over the last objective; the base case of two objectives is a sweep.
fn slice_volume(mut points: Vec<Vec<f64>>, reference: &[f64]) -> f64 {
    let d = reference.len();
    points = filter_non_dominated(points);
    if points.is_empty() {
        return 0.0;
    }

    match d {
        1 => reference[0] - points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
        2 => sweep_2d(points, reference),
        _ => {
            points.sort_by(|a, b| a[d - 1].total_cmp(&b[d - 1]));
            let mut volume = 0.0;
            for i in 0..points.len() {
                let upper = points
                    .get(i + 1)
                    .map_or(reference[d - 1], |next| next[d - 1]);
                let depth = upper - points[i][d - 1];
                if depth <= 0.0 {
                    continue;
                }

                let slice: Vec<Vec<f64>> =
                    points[..=i].iter().map(|p| p[..d - 1].to_vec()).collect();
                volume += depth * slice_volume(slice, &reference[..d - 1]);
            }

            volume
        }
    }
}

/// The area dominated by a two-dimensional point set.
fn sweep_2d(mut points: Vec<Vec<f64>>, reference: &[f64]) -> f64 {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let mut area = 0.0;
    let mut ceiling = reference[1];
    for p in &points {
        if p[1] < ceiling {
            area += (reference[0] - p[0]) * (ceiling - p[1]);
            ceiling = p[1];
        }
    }

    area
}

/// Removes dominated points and duplicates.
fn filter_non_dominated(points: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut kept: Vec<Vec<f64>> = Vec::with_capacity(points.len());
    for p in points {
        if kept.iter().any(|q| dominates(q, &p) || *q == p) {
            continue;
        }

        kept.retain(|q| !dominates(&p, q));
        kept.push(p);
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed volumes in one, two and three dimensions, and the cube corner case.
    #[test]
    fn known_volumes() {
        assert_eq!(hypervolume(&[[2.0]], &[5.0]), 3.0);
        assert_eq!(
            hypervolume(&[[1.0, 1.0], [2.0, 0.5]], &[3.0, 3.0]),
            4.0 + 0.5
        );
        assert_eq!(hypervolume(&[[0.0, 0.0, 0.0]], &[1.0, 2.0, 3.0]), 6.0);

        // Two boxes [0,2]x[0,1]x[0,1] and [0,1]x[0,2]x[0,1] in the box to (2,2,2), overlap 1.
        let points = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]];
        assert!((hypervolume(&points, &[2.0, 2.0, 2.0]) - 3.0).abs() < 1e-12);

        // Points outside the reference box contribute nothing.
        assert_eq!(hypervolume(&[[5.0, 0.0]], &[3.0, 3.0]), 0.0);
    }

    /// The 3D volume equals a Monte-Carlo-free grid count on integer points.
    #[test]
    fn matches_grid_count() {
        let points = [
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
        ];
        let reference = [3.0, 3.0, 3.0];
        let mut count = 0;
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    let cell = [f64::from(x), f64::from(y), f64::from(z)];
                    if points
                        .iter()
                        .any(|p| p.iter().zip(&cell).all(|(a, c)| a <= c))
                    {
                        count += 1;
                    }
                }
            }
        }

        assert!((hypervolume(&points, &reference) - f64::from(count)).abs() < 1e-12);
    }
}
