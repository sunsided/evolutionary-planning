//! Structured reference points and weight vectors on the unit simplex (Das and Dennis 1998),
//! used by NSGA-III and MOEA/D.

use alloc::vec::Vec;

/// All points of the simplex `{w ≥ 0, Σ w = 1}` in `m` dimensions whose coordinates are
/// multiples of `1/p`: `C(m + p - 1, p)` points.
///
/// ```
/// use evolutionary_planning::pareto::das_dennis;
///
/// assert_eq!(das_dennis(3, 12).len(), 91);
/// assert_eq!(das_dennis(2, 4), vec![vec![0.0, 1.0], vec![0.25, 0.75], vec![0.5, 0.5], vec![0.75, 0.25], vec![1.0, 0.0]]);
/// ```
///
/// # Panics
///
/// Panics if `m == 0` or `p == 0`.
#[must_use]
pub fn das_dennis(m: usize, p: usize) -> Vec<Vec<f64>> {
    assert!(m > 0 && p > 0, "das_dennis needs m > 0 and p > 0");

    let mut points = Vec::new();
    let mut current = alloc::vec![0usize; m];
    compositions(&mut current, 0, p, &mut |c| {
        #[expect(clippy::cast_precision_loss, reason = "divisions are tiny integers")]
        let point = c.iter().map(|&k| k as f64 / p as f64).collect();
        points.push(point);
    });

    points
}

/// Enumerates the compositions of `remaining` into `current[index..]`, first coordinate
/// ascending.
fn compositions(
    current: &mut [usize],
    index: usize,
    remaining: usize,
    emit: &mut impl FnMut(&[usize]),
) {
    if index == current.len() - 1 {
        current[index] = remaining;
        emit(current);
        return;
    }

    for k in 0..=remaining {
        current[index] = k;
        compositions(current, index + 1, remaining - k, emit);
    }
}

/// Two layers of Das-Dennis points (Deb and Jain 2014) for many objectives: an outer layer
/// with `outer` divisions and an inner one with `inner` divisions shrunk halfway towards the
/// simplex centre.
///
/// # Panics
///
/// Panics if `m == 0` or a division count is `0`.
#[must_use]
pub fn two_layer(m: usize, outer: usize, inner: usize) -> Vec<Vec<f64>> {
    #[expect(clippy::cast_precision_loss, reason = "objective counts are tiny")]
    let centre = 1.0 / m as f64;
    let mut points = das_dennis(m, outer);
    points.extend(
        das_dennis(m, inner)
            .into_iter()
            .map(|p| p.into_iter().map(|w| 0.5 * w + 0.5 * centre).collect()),
    );

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every point is on the simplex and the counts are binomial coefficients.
    #[test]
    fn simplex_points() {
        for (m, p, count) in [(2, 5, 6), (3, 4, 15), (5, 3, 35)] {
            let points = das_dennis(m, p);
            assert_eq!(points.len(), count);
            assert!(
                points
                    .iter()
                    .all(|w| (w.iter().sum::<f64>() - 1.0).abs() < 1e-12)
            );
        }

        let layered = two_layer(3, 3, 2);
        assert_eq!(layered.len(), 10 + 6);
        assert!(
            layered
                .iter()
                .all(|w| (w.iter().sum::<f64>() - 1.0).abs() < 1e-12)
        );
    }
}
