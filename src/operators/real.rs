//! Operators for real vectors `Vec<f64>`: box [`Bounds`], uniform and Latin hypercube
//! initialisation, Gaussian, polynomial and uniform-reset mutation, SBX, BLX-α and arithmetic
//! crossover, and clamping or reflecting repair.
//!
//! Following Jakob's rule that explicit bounds belong in the operators, every bounded operator
//! keeps its children inside the box; Gaussian mutation is unbounded and pairs with
//! [`Clamp`] or [`Reflect`].

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Crossover, Initializer, Mutation, Repair};
use crate::rng::{below, chance, shuffle, standard_normal, uniform, unit_f64};

/// An axis-aligned box `[lower_i, upper_i]` per coordinate.
///
/// ```
/// use evolutionary_planning::operators::{Initializer, real::Bounds};
/// use evolutionary_planning::SplitMix64;
///
/// let bounds = Bounds::uniform(3, -5.0, 5.0);
/// let x = bounds.clone().initialize(&mut SplitMix64::new(1));
/// assert!(bounds.contains(&x));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    /// Lower bounds, one per coordinate; finite, `lower[i] <= upper[i]`.
    lower: Vec<f64>,
    /// Upper bounds, one per coordinate; finite.
    upper: Vec<f64>,
}

impl Bounds {
    /// Creates a box from per-coordinate bounds.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ, a bound is not finite, or `lower[i] > upper[i]`.
    #[must_use]
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        assert_eq!(lower.len(), upper.len(), "bounds of different dimension");
        for (lo, hi) in lower.iter().zip(&upper) {
            assert!(
                lo.is_finite() && hi.is_finite() && lo <= hi,
                "invalid bound [{lo}, {hi}]"
            );
        }

        Self { lower, upper }
    }

    /// The box `[low, high]^dimension`.
    ///
    /// # Panics
    ///
    /// Panics if `low > high` or either is not finite.
    #[must_use]
    pub fn uniform(dimension: usize, low: f64, high: f64) -> Self {
        Self::new(alloc::vec![low; dimension], alloc::vec![high; dimension])
    }

    /// The number of coordinates.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.lower.len()
    }

    /// The lower bounds.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// The upper bounds.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// The width `upper[i] - lower[i]` of coordinate `i`.
    #[must_use]
    pub fn width(&self, i: usize) -> f64 {
        self.upper[i] - self.lower[i]
    }

    /// Returns `true` if `x` has the right dimension and lies inside the box.
    #[must_use]
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dimension()
            && x.iter()
                .enumerate()
                .all(|(i, v)| (self.lower[i]..=self.upper[i]).contains(v))
    }

    /// Clamps `x` into the box coordinate by coordinate.
    pub fn clamp(&self, x: &mut [f64]) {
        for (i, v) in x.iter_mut().enumerate() {
            *v = v.clamp(self.lower[i], self.upper[i]);
        }
    }

    /// The centre of the box.
    #[must_use]
    pub fn center(&self) -> Vec<f64> {
        self.lower
            .iter()
            .zip(&self.upper)
            .map(|(lo, hi)| 0.5 * (lo + hi))
            .collect()
    }
}

/// Uniform random points in the box.
impl Initializer<Vec<f64>> for Bounds {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<f64> {
        (0..self.dimension())
            .map(|i| uniform(rng, self.lower[i], self.upper[i]))
            .collect()
    }
}

/// Returns `count` points of a Latin hypercube sample of `bounds`: every coordinate's range is
/// split into `count` equal strata, each stratum holds exactly one point per coordinate, and
/// the strata are permuted independently per coordinate (McKay, Beckman and Conover 1979).
///
/// Yue et al. (2023) initialise DE this way for more uniform coverage with few points.
///
/// ```
/// use evolutionary_planning::operators::real::{Bounds, latin_hypercube};
/// use evolutionary_planning::SplitMix64;
///
/// let points = latin_hypercube(&Bounds::uniform(2, 0.0, 1.0), 4, &mut SplitMix64::new(1));
/// let mut strata: Vec<usize> = points.iter().map(|p| (p[0] * 4.0) as usize).collect();
/// strata.sort();
/// assert_eq!(strata, [0, 1, 2, 3]);
/// ```
pub fn latin_hypercube<R: Rng + ?Sized>(
    bounds: &Bounds,
    count: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let mut points = alloc::vec![alloc::vec![0.0; bounds.dimension()]; count];
    let mut strata: Vec<usize> = (0..count).collect();
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample counts are far below 2^52"
    )]
    let n = count as f64;
    for i in 0..bounds.dimension() {
        shuffle(rng, &mut strata);
        for (point, &stratum) in points.iter_mut().zip(&strata) {
            #[expect(
                clippy::cast_precision_loss,
                reason = "sample counts are far below 2^52"
            )]
            let offset = (stratum as f64 + unit_f64(rng)) / n;
            point[i] = (bounds.lower[i] + offset * bounds.width(i)).min(bounds.upper[i]);
        }
    }

    points
}

/// Returns the per-gene mutation probability: `rate`, or `1 / n` if `None`.
fn gene_rate(rate: Option<f64>, n: usize) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
    let default = 1.0 / n.max(1) as f64;

    rate.unwrap_or(default)
}

/// Adds `N(0, sigma^2)` to each coordinate with probability `rate` (default `1/n`); at least
/// one coordinate changes. Unbounded; combine with [`Clamp`] or [`Reflect`] if needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    /// The standard deviation, in the units of the coordinates.
    pub sigma: f64,
    /// The per-coordinate probability; `None` means `1/n`.
    pub rate: Option<f64>,
}

impl Mutation<Vec<f64>> for Gaussian {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<f64>, rng: &mut R) {
        let rate = gene_rate(self.rate, genome.len());
        for_some_genes(genome.len(), rate, rng, |i, rng| {
            genome[i] += self.sigma * standard_normal(rng);
        });
    }
}

/// Calls `f` for each index with probability `rate`, and for one uniform index if no other was
/// chosen, so a mutation never silently returns its input unchanged.
fn for_some_genes<R: Rng + ?Sized>(
    n: usize,
    rate: f64,
    rng: &mut R,
    mut f: impl FnMut(usize, &mut R),
) {
    if n == 0 {
        return;
    }

    let mut any = false;
    for i in 0..n {
        if chance(rng, rate) {
            f(i, rng);
            any = true;
        }
    }

    if !any {
        let i = below(rng, n);
        f(i, rng);
    }
}

/// Deb's bounded polynomial mutation with distribution index `eta` (Deb and Goyal 1996, in
/// the form of the NSGA-II reference code); each coordinate mutates with probability `rate`
/// (default `1/n`), at least one does.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    /// The box the children stay in.
    pub bounds: Bounds,
    /// The distribution index; larger values give smaller steps. `20` is customary.
    pub eta: f64,
    /// The per-coordinate probability; `None` means `1/n`.
    pub rate: Option<f64>,
}

impl Mutation<Vec<f64>> for Polynomial {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<f64>, rng: &mut R) {
        let rate = gene_rate(self.rate, genome.len());
        let bounds = &self.bounds;
        let eta = self.eta;
        for_some_genes(genome.len(), rate, rng, |i, rng| {
            genome[i] = polynomial_step(
                genome[i],
                bounds.lower[i],
                bounds.upper[i],
                eta,
                unit_f64(rng),
            );
        });
    }
}

/// One polynomial-mutation step of `y` in `[low, high]` for the uniform draw `u`.
fn polynomial_step(y: f64, low: f64, high: f64, eta: f64, u: f64) -> f64 {
    let width = high - low;
    if width <= 0.0 {
        return low;
    }

    let power = 1.0 / (eta + 1.0);
    let delta_q = if u <= 0.5 {
        let xy = 1.0 - (y - low) / width;
        let value = 2.0 * u + (1.0 - 2.0 * u) * libm::pow(xy, eta + 1.0);
        libm::pow(value, power) - 1.0
    } else {
        let xy = 1.0 - (high - y) / width;
        let value = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * libm::pow(xy, eta + 1.0);
        1.0 - libm::pow(value, power)
    };

    (y + delta_q * width).clamp(low, high)
}

/// Replaces each coordinate with probability `rate` (default `1/n`) by a uniform value in its
/// bounds; at least one coordinate changes.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformReset {
    /// The box new values are drawn from.
    pub bounds: Bounds,
    /// The per-coordinate probability; `None` means `1/n`.
    pub rate: Option<f64>,
}

impl Mutation<Vec<f64>> for UniformReset {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<f64>, rng: &mut R) {
        let rate = gene_rate(self.rate, genome.len());
        let bounds = &self.bounds;
        for_some_genes(genome.len(), rate, rng, |i, rng| {
            genome[i] = uniform(rng, bounds.lower[i], bounds.upper[i]);
        });
    }
}

/// Simulated binary crossover (Deb and Agrawal 1995), bounded form of the NSGA-II reference
/// code: each coordinate is recombined with probability `1/2`, the children's spread follows
/// the distribution index `eta`, and the children are swapped per coordinate with probability
/// `1/2`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sbx {
    /// The box the children stay in.
    pub bounds: Bounds,
    /// The distribution index; larger values keep children closer to the parents. `15`–`20`
    /// is customary.
    pub eta: f64,
}

impl Crossover<Vec<f64>> for Sbx {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<f64>,
        b: &Vec<f64>,
        rng: &mut R,
    ) -> [Vec<f64>; 2] {
        let mut c1 = a.clone();
        let mut c2 = b.clone();
        for i in 0..a.len().min(b.len()) {
            if unit_f64(rng) > 0.5 || (a[i] - b[i]).abs() <= 1e-14 {
                continue;
            }

            let (low, high) = (self.bounds.lower[i], self.bounds.upper[i]);
            let (y1, y2) = if a[i] < b[i] {
                (a[i], b[i])
            } else {
                (b[i], a[i])
            };
            let u = unit_f64(rng);
            let lower_child = sbx_child(
                y1,
                y2,
                1.0 + 2.0 * (y1 - low) / (y2 - y1),
                self.eta,
                u,
                -1.0,
            );
            let upper_child = sbx_child(
                y1,
                y2,
                1.0 + 2.0 * (high - y2) / (y2 - y1),
                self.eta,
                u,
                1.0,
            );
            let (x1, x2) = (lower_child.clamp(low, high), upper_child.clamp(low, high));
            if unit_f64(rng) < 0.5 {
                c1[i] = x2;
                c2[i] = x1;
            } else {
                c1[i] = x1;
                c2[i] = x2;
            }
        }

        [c1, c2]
    }
}

/// One SBX child of the sorted parents `y1 < y2` given the bound ratio `beta`, the uniform draw
/// `u`, and `sign` (`-1` towards the lower bound, `+1` towards the upper).
fn sbx_child(y1: f64, y2: f64, beta: f64, eta: f64, u: f64, sign: f64) -> f64 {
    let alpha = 2.0 - libm::pow(beta, -(eta + 1.0));
    let beta_q = if u <= 1.0 / alpha {
        libm::pow(u * alpha, 1.0 / (eta + 1.0))
    } else {
        libm::pow(1.0 / (2.0 - u * alpha), 1.0 / (eta + 1.0))
    };

    0.5 * ((y1 + y2) + sign * beta_q * (y2 - y1))
}

/// Blend crossover BLX-α (Eshelman and Schaffer 1993): each child coordinate is uniform in
/// the parents' interval widened by `alpha` times its length on both sides, clamped to the
/// bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct Blx {
    /// The box the children stay in.
    pub bounds: Bounds,
    /// The widening factor; `0.5` is customary.
    pub alpha: f64,
}

impl Crossover<Vec<f64>> for Blx {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<f64>,
        b: &Vec<f64>,
        rng: &mut R,
    ) -> [Vec<f64>; 2] {
        let mut children = [a.clone(), b.clone()];
        for child in &mut children {
            for i in 0..a.len().min(b.len()) {
                let (lo, hi) = (a[i].min(b[i]), a[i].max(b[i]));
                let spread = self.alpha * (hi - lo);
                let value = uniform(rng, lo - spread, hi + spread);
                child[i] = value.clamp(self.bounds.lower[i], self.bounds.upper[i]);
            }
        }

        children
    }
}

/// Whole arithmetic crossover: `λa + (1-λ)b` and `(1-λ)a + λb` with one uniform `λ`. Children
/// stay in the parents' bounding box, hence in any convex feasible set containing both.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Arithmetic;

impl Crossover<Vec<f64>> for Arithmetic {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<f64>,
        b: &Vec<f64>,
        rng: &mut R,
    ) -> [Vec<f64>; 2] {
        let lambda = unit_f64(rng);
        let blend = |x: f64, y: f64, l: f64| l * x + (1.0 - l) * y;

        [
            a.iter()
                .zip(b)
                .map(|(&x, &y)| blend(x, y, lambda))
                .collect(),
            a.iter()
                .zip(b)
                .map(|(&x, &y)| blend(x, y, 1.0 - lambda))
                .collect(),
        ]
    }
}

/// Iso+line variation (Vassiliades and Mouret 2018), the directional crossover of
/// MAP-Elites: `a + σ_iso N(0, I) + σ_line N(0, 1) (b - a)`, and symmetrically for the second
/// child. Unbounded; combine with [`Clamp`] or [`Reflect`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsoLineDd {
    /// The isotropic Gaussian spread `σ_iso`.
    pub sigma_iso: f64,
    /// The spread `σ_line` along the line between the parents.
    pub sigma_line: f64,
}

impl Crossover<Vec<f64>> for IsoLineDd {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<f64>,
        b: &Vec<f64>,
        rng: &mut R,
    ) -> [Vec<f64>; 2] {
        let child = |x: &[f64], y: &[f64], rng: &mut R| {
            let line = self.sigma_line * standard_normal(rng);
            x.iter()
                .zip(y)
                .map(|(p, q)| p + self.sigma_iso * standard_normal(rng) + line * (q - p))
                .collect()
        };

        [child(a, b, rng), child(b, a, rng)]
    }
}

/// Clamps every coordinate into the bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct Clamp(pub Bounds);

impl Repair<Vec<f64>> for Clamp {
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut Vec<f64>, _rng: &mut R) {
        self.0.clamp(genome);
    }
}

/// Reflects coordinates that left the box back in at the violated bound (mirror boundary),
/// which, unlike clamping, does not pile up probability mass on the boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct Reflect(pub Bounds);

impl Repair<Vec<f64>> for Reflect {
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut Vec<f64>, _rng: &mut R) {
        for (i, value) in genome.iter_mut().enumerate() {
            *value = reflect(*value, self.0.lower[i], self.0.upper[i]);
        }
    }
}

/// Folds `x` into `[low, high]` by repeated mirroring at the bounds.
fn reflect(x: f64, low: f64, high: f64) -> f64 {
    let width = high - low;
    if width <= 0.0 || !x.is_finite() {
        return x.clamp(low, high);
    }

    let period = 2.0 * width;
    let mut offset = (x - low) % period;
    if offset < 0.0 {
        offset += period;
    }

    if offset <= width {
        low + offset
    } else {
        high - (offset - width)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::SplitMix64;

    /// Bounded operators keep their children in the box.
    #[test]
    fn bounded_operators_stay_inside() {
        let bounds = Bounds::uniform(5, -1.0, 2.0);
        let mut rng = SplitMix64::new(7);
        let mut init = bounds.clone();
        let mut sbx = Sbx {
            bounds: bounds.clone(),
            eta: 15.0,
        };
        let mut blx = Blx {
            bounds: bounds.clone(),
            alpha: 0.5,
        };
        let mut poly = Polynomial {
            bounds: bounds.clone(),
            eta: 20.0,
            rate: Some(1.0),
        };
        for _ in 0..500 {
            let a = init.initialize(&mut rng);
            let b = init.initialize(&mut rng);
            for mut child in sbx
                .crossover(&a, &b, &mut rng)
                .into_iter()
                .chain(blx.crossover(&a, &b, &mut rng))
            {
                assert!(bounds.contains(&child), "{child:?}");
                poly.mutate(&mut child, &mut rng);
                assert!(bounds.contains(&child), "{child:?}");
            }
        }
    }

    /// Reflection folds values back and is the identity inside.
    #[test]
    fn reflection_folds() {
        assert!((reflect(2.5, 0.0, 2.0) - 1.5).abs() < 1e-12);
        assert!((reflect(-0.5, 0.0, 2.0) - 0.5).abs() < 1e-12);
        assert!((reflect(5.0, 0.0, 2.0) - 1.0).abs() < 1e-12);
        assert!((reflect(1.25, 0.0, 2.0) - 1.25).abs() < 1e-12);
    }

    /// Identical SBX parents give identical children.
    #[test]
    fn sbx_of_equal_parents_is_identity() {
        let mut sbx = Sbx {
            bounds: Bounds::uniform(3, 0.0, 1.0),
            eta: 15.0,
        };
        let p = vec![0.2, 0.5, 0.9];

        assert_eq!(
            sbx.crossover(&p, &p, &mut SplitMix64::new(1)),
            [p.clone(), p]
        );
    }

    /// Latin hypercube samples hit every stratum of every coordinate exactly once.
    #[test]
    fn latin_hypercube_stratifies() {
        let bounds = Bounds::uniform(3, -2.0, 2.0);
        let points = latin_hypercube(&bounds, 10, &mut SplitMix64::new(3));
        for i in 0..3 {
            let mut strata: Vec<usize> = points
                .iter()
                .map(|p| {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "test"
                    )]
                    let s = ((p[i] + 2.0) / 4.0 * 10.0) as usize;
                    s
                })
                .collect();
            strata.sort_unstable();
            assert_eq!(strata, (0..10).collect::<Vec<_>>());
        }
    }
}
