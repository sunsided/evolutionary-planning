//! Multi-objective benchmark problems: Schaffer, ZDT1–3, DTLZ1–2 and a bi-objective 0/1
//! knapsack with a capacity constraint. All objectives are minimised.

use alloc::vec::Vec;

use crate::MultiObjective;

/// Schaffer's problem N.1 on one real `x`: `(x², (x - 2)²)`; the Pareto set is `[0, 2]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Schaffer;

impl MultiObjective<Vec<f64>> for Schaffer {
    fn objectives(&self) -> usize {
        2
    }

    fn evaluate(&mut self, genome: &Vec<f64>, out: &mut [f64]) -> f64 {
        let x = genome[0];
        out[0] = x * x;
        out[1] = (x - 2.0) * (x - 2.0);

        0.0
    }
}

/// The ZDT problems of Zitzler, Deb and Thiele (2000) on `[0, 1]^n`, two objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zdt {
    /// Convex front `f2 = 1 - sqrt(f1)`.
    Zdt1,
    /// Concave front `f2 = 1 - f1²`.
    Zdt2,
    /// Disconnected front `f2 = 1 - sqrt(f1) - f1 sin(10π f1)`.
    Zdt3,
}

impl Zdt {
    /// The optimal `f2` for `f1` on the front (where it is Pareto-optimal: everywhere for
    /// ZDT1 and ZDT2, on five disjoint pieces for ZDT3).
    #[must_use]
    pub fn front(self, f1: f64) -> f64 {
        self.shape(f1, 1.0)
    }

    /// `f2 / g` as a function of `f1 / g`, times `g`.
    fn shape(self, f1: f64, g: f64) -> f64 {
        let ratio = f1 / g;
        let h = match self {
            Self::Zdt1 => 1.0 - libm::sqrt(ratio),
            Self::Zdt2 => 1.0 - ratio * ratio,
            Self::Zdt3 => {
                1.0 - libm::sqrt(ratio) - ratio * libm::sin(10.0 * core::f64::consts::PI * f1)
            }
        };

        g * h
    }
}

impl MultiObjective<Vec<f64>> for Zdt {
    fn objectives(&self) -> usize {
        2
    }

    fn evaluate(&mut self, genome: &Vec<f64>, out: &mut [f64]) -> f64 {
        let n = genome.len();
        #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
        let g = if n > 1 {
            1.0 + 9.0 * genome[1..].iter().sum::<f64>() / (n - 1) as f64
        } else {
            1.0
        };
        out[0] = genome[0];
        out[1] = self.shape(genome[0], g);

        0.0
    }
}

/// The DTLZ problems of Deb, Thiele, Laumanns and Zitzler (2002) with `objectives` objectives
/// on `[0, 1]^n`, `n >= objectives`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dtlz {
    /// Which problem.
    pub kind: DtlzKind,
    /// The number of objectives `m >= 2`.
    pub objectives: usize,
}

/// The DTLZ variants implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtlzKind {
    /// Linear front `Σ f_i = 0.5`, with a multimodal distance function.
    Dtlz1,
    /// Spherical front `Σ f_i² = 1`.
    Dtlz2,
}

impl MultiObjective<Vec<f64>> for Dtlz {
    fn objectives(&self) -> usize {
        self.objectives
    }

    fn evaluate(&mut self, x: &Vec<f64>, out: &mut [f64]) -> f64 {
        let m = self.objectives;
        let tail = &x[m - 1..];
        let half_pi = core::f64::consts::FRAC_PI_2;
        match self.kind {
            DtlzKind::Dtlz1 => {
                #[expect(clippy::cast_precision_loss, reason = "dimensions are far below 2^52")]
                let k = tail.len() as f64;
                let g = 100.0
                    * (k + tail
                        .iter()
                        .map(|v| {
                            (v - 0.5) * (v - 0.5)
                                - libm::cos(20.0 * core::f64::consts::PI * (v - 0.5))
                        })
                        .sum::<f64>());
                for (i, f) in out.iter_mut().enumerate().take(m) {
                    let mut value = f64::midpoint(1.0, g);
                    value *= x[..m - 1 - i].iter().product::<f64>();
                    if i > 0 {
                        value *= 1.0 - x[m - 1 - i];
                    }
                    *f = value;
                }
            }
            DtlzKind::Dtlz2 => {
                let g: f64 = tail.iter().map(|v| (v - 0.5) * (v - 0.5)).sum();
                for (i, f) in out.iter_mut().enumerate().take(m) {
                    let mut value = 1.0 + g;
                    value *= x[..m - 1 - i]
                        .iter()
                        .map(|v| libm::cos(v * half_pi))
                        .product::<f64>();
                    if i > 0 {
                        value *= libm::sin(x[m - 1 - i] * half_pi);
                    }
                    *f = value;
                }
            }
        }

        0.0
    }
}

/// A bi-objective 0/1 knapsack: maximise two values (returned negated as costs) under a weight
/// capacity; the violation is the excess weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Knapsack {
    /// Per item: `(weight, value_1, value_2)`.
    pub items: Vec<(u32, u32, u32)>,
    /// The weight capacity.
    pub capacity: u32,
}

impl Knapsack {
    /// A deterministic pseudo-random instance with `n` items and capacity half the total
    /// weight; weights and values are in `1..=30`.
    ///
    /// # Panics
    ///
    /// Panics if the total weight overflows `u32` (more than about `10^8` items).
    #[must_use]
    pub fn generate(n: usize, seed: u64) -> Self {
        let mut rng = crate::SplitMix64::new(seed);
        let mut draw = || 1 + u32::try_from(crate::below(&mut rng, 30)).expect("below 30");
        let items: Vec<(u32, u32, u32)> = (0..n).map(|_| (draw(), draw(), draw())).collect();
        let capacity = items.iter().map(|i| i.0).sum::<u32>() / 2;

        Self { items, capacity }
    }

    /// The costs `(-value_1, -value_2)` and the excess weight of a selection.
    #[must_use]
    pub fn assess(&self, take: &[bool]) -> ([f64; 2], f64) {
        let (mut weight, mut v1, mut v2) = (0, 0, 0);
        for (&(w, a, b), _) in self.items.iter().zip(take).filter(|(_, t)| **t) {
            weight += w;
            v1 += a;
            v2 += b;
        }

        (
            [-f64::from(v1), -f64::from(v2)],
            f64::from(weight.saturating_sub(self.capacity)),
        )
    }
}

impl MultiObjective<Vec<bool>> for Knapsack {
    fn objectives(&self) -> usize {
        2
    }

    fn evaluate(&mut self, genome: &Vec<bool>, out: &mut [f64]) -> f64 {
        let (costs, violation) = self.assess(genome);
        out.copy_from_slice(&costs);

        violation
    }
}

/// LOTZ, LeadingOnes-TrailingZeros (Laumanns, Thiele and Zitzler 2004), as costs: minimise
/// `-(leading ones)` and `-(trailing zeros)` of a bit string of length `n`. The Pareto front
/// is the `n + 1` points `(-k, -(n - k))`, each reached by `1^k 0^(n-k)` only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lotz;

impl MultiObjective<Vec<bool>> for Lotz {
    fn objectives(&self) -> usize {
        2
    }

    fn evaluate(&mut self, genome: &Vec<bool>, out: &mut [f64]) -> f64 {
        let leading = genome.iter().take_while(|&&b| b).count();
        let trailing = genome.iter().rev().take_while(|&&b| !b).count();
        #[expect(clippy::cast_precision_loss, reason = "lengths are far below 2^52")]
        {
            out[0] = -(leading as f64);
            out[1] = -(trailing as f64);
        }

        0.0
    }
}

/// Every objective vector (and violation) of the `2^n` bit strings of length `n`, for
/// brute-force reference fronts.
///
/// # Panics
///
/// Panics if `n > 20` (a million genomes).
#[must_use]
pub fn enumerate_bits<M: MultiObjective<Vec<bool>>>(
    problem: &mut M,
    n: usize,
) -> Vec<(Vec<bool>, Vec<f64>, f64)> {
    assert!(n <= 20, "too many genomes to enumerate");

    let m = problem.objectives();
    (0..1u32 << n)
        .map(|mask| {
            let genome: Vec<bool> = (0..n).map(|i| mask >> i & 1 == 1).collect();
            let mut out = alloc::vec![0.0; m];
            let violation = problem.evaluate(&genome, &mut out);
            (genome, out, violation)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points of the Pareto set lie on the documented fronts.
    #[test]
    fn fronts() {
        let mut out = [0.0; 3];
        let mut zdt = Zdt::Zdt1;
        zdt.evaluate(&alloc::vec![0.25, 0.0, 0.0], &mut out[..2]);
        assert!((out[1] - 0.5).abs() < 1e-12);

        let mut dtlz2 = Dtlz {
            kind: DtlzKind::Dtlz2,
            objectives: 3,
        };
        dtlz2.evaluate(&alloc::vec![0.3, 0.7, 0.5, 0.5], &mut out);
        assert!((out.iter().map(|f| f * f).sum::<f64>() - 1.0).abs() < 1e-12);

        let mut dtlz1 = Dtlz {
            kind: DtlzKind::Dtlz1,
            objectives: 3,
        };
        dtlz1.evaluate(&alloc::vec![0.3, 0.7, 0.5, 0.5], &mut out);
        assert!((out.iter().sum::<f64>() - 0.5).abs() < 1e-12);
    }
}
