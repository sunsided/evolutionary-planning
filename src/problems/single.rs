//! Single-objective benchmark functions, all minimised with optimum `0` unless noted.

use alloc::vec::Vec;

/// The sphere `Σ x_i²`; optimum `0` at the origin.
#[must_use]
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

/// The axis-parallel ellipsoid `Σ 10^(6 (i-1)/(n-1)) x_i²` with condition number `10^6`, as
/// in Hansen's tutorial; optimum `0` at the origin. For `n = 1` it is the sphere.
#[must_use]
pub fn ellipsoid(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return sphere(x);
    }

    #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
    let denominator = (n - 1) as f64;
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
            let exponent = 6.0 * i as f64 / denominator;
            libm::pow(10.0, exponent) * v * v
        })
        .sum()
}

/// Rosenbrock's function `Σ 100 (x_{i+1} - x_i²)² + (1 - x_i)²`; optimum `0` at
/// `(1, …, 1)`. For `n >= 4` it has a second local minimum near `(-1, 1, …, 1)`.
#[must_use]
pub fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| {
            let (a, b) = (w[1] - w[0] * w[0], 1.0 - w[0]);
            100.0 * a * a + b * b
        })
        .sum()
}

/// Rastrigin's function `10 n + Σ (x_i² - 10 cos(2π x_i))`; optimum `0` at the origin,
/// highly multimodal.
#[must_use]
pub fn rastrigin(x: &[f64]) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
    let base = 10.0 * x.len() as f64;

    base + x
        .iter()
        .map(|v| v * v - 10.0 * libm::cos(2.0 * core::f64::consts::PI * v))
        .sum::<f64>()
}

/// OneMax as a cost: the number of zero bits; optimum `0` (all ones).
#[must_use]
pub fn one_max(bits: &[bool]) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "lengths are far below 2^52")]
    let zeros = bits.iter().filter(|&&b| !b).count() as f64;

    zeros
}

/// LeadingOnes as a cost: `n` minus the length of the leading run of ones; optimum `0`.
#[must_use]
pub fn leading_ones(bits: &[bool]) -> f64 {
    let ones = bits.iter().take_while(|&&b| b).count();

    #[expect(clippy::cast_precision_loss, reason = "lengths are far below 2^52")]
    let cost = (bits.len() - ones) as f64;

    cost
}

/// A symmetric travelling-salesperson instance over points in the plane.
#[derive(Debug, Clone, PartialEq)]
pub struct Tour {
    /// The city coordinates.
    pub cities: Vec<(f64, f64)>,
}

impl Tour {
    /// The length of the closed tour visiting the cities in `order`.
    #[must_use]
    pub fn length(&self, order: &[usize]) -> f64 {
        let n = order.len();
        (0..n)
            .map(|k| {
                let (a, b) = (self.cities[order[k]], self.cities[order[(k + 1) % n]]);
                libm::hypot(a.0 - b.0, a.1 - b.1)
            })
            .sum()
    }

    /// The `n` cities on a circle of radius 1, numbered in a scrambled order; the optimal tour
    /// is the circle, of length `2 n sin(π / n)`.
    #[must_use]
    pub fn circle(n: usize) -> Self {
        let cities = (0..n)
            .map(|k| {
                // A fixed scramble (7k mod n with n coprime to 7 would also do); any order works.
                #[expect(clippy::cast_precision_loss, reason = "city counts are tiny")]
                let angle = 2.0 * core::f64::consts::PI * ((k * 7) % n) as f64 / n as f64;
                (libm::cos(angle), libm::sin(angle))
            })
            .collect();

        Self { cities }
    }

    /// The optimal length of [`Tour::circle`]`(n)`.
    #[must_use]
    pub fn circle_optimum(n: usize) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "city counts are tiny")]
        let n = n as f64;

        2.0 * n * libm::sin(core::f64::consts::PI / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The optima are where they are documented.
    #[test]
    fn optima() {
        assert_eq!(sphere(&[0.0; 5]), 0.0);
        assert_eq!(ellipsoid(&[0.0; 5]), 0.0);
        assert_eq!(rosenbrock(&[1.0; 5]), 0.0);
        assert!(rastrigin(&[0.0; 5]).abs() < 1e-12);
        assert_eq!(one_max(&[true; 4]), 0.0);
        assert_eq!(leading_ones(&[true, false, true]), 2.0);
        assert!((ellipsoid(&[0.0, 1.0]) - 1e6).abs() < 1e-6);
    }

    /// The circle tour in angular order has the documented length.
    #[test]
    fn circle_tour_optimum() {
        let tour = Tour::circle(11);
        let mut order: alloc::vec::Vec<usize> = (0..11).collect();
        order.sort_by(|&a, &b| {
            let angle = |k: usize| libm::atan2(tour.cities[k].1, tour.cities[k].0);
            angle(a).total_cmp(&angle(b))
        });

        assert!((tour.length(&order) - Tour::circle_optimum(11)).abs() < 1e-9);
    }
}
