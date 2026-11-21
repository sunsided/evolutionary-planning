//! Positional crossovers for any `Vec<T>`: one-point, two-point and uniform. They also accept
//! parents of different lengths: cut points lie within the shorter parent and tails are
//! exchanged whole, so lengths are swapped, never invented.

use alloc::vec::Vec;

use rand_core::Rng;

use super::Crossover;
use crate::rng::{below, two_sorted_indices, unit_f64};

/// One-point crossover: cut both parents at the same position `c ∈ 1..min_len` and exchange
/// the tails.
///
/// ```
/// use evolutionary_planning::operators::{Crossover, sequence::OnePoint};
/// use evolutionary_planning::SplitMix64;
///
/// let [c1, c2] = OnePoint.crossover(&vec![0; 4], &vec![1; 4], &mut SplitMix64::new(1));
/// assert_eq!(c1.len() + c2.len(), 8);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OnePoint;

impl<T: Clone> Crossover<Vec<T>> for OnePoint {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &Vec<T>, b: &Vec<T>, rng: &mut R) -> [Vec<T>; 2] {
        let shorter = a.len().min(b.len());
        if shorter < 2 {
            return [a.clone(), b.clone()];
        }

        let cut = 1 + below(rng, shorter - 1);

        [splice(a, b, cut, cut), splice(b, a, cut, cut)]
    }
}

/// Returns `head[..head_cut]` followed by `tail[tail_cut..]`.
pub(crate) fn splice<T: Clone>(head: &[T], tail: &[T], head_cut: usize, tail_cut: usize) -> Vec<T> {
    let mut child = Vec::with_capacity(head_cut + tail.len() - tail_cut);
    child.extend_from_slice(&head[..head_cut]);
    child.extend_from_slice(&tail[tail_cut..]);

    child
}

/// Two-point crossover: exchange the segment between two cut points within the shorter
/// parent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TwoPoint;

impl<T: Clone> Crossover<Vec<T>> for TwoPoint {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &Vec<T>, b: &Vec<T>, rng: &mut R) -> [Vec<T>; 2] {
        let shorter = a.len().min(b.len());
        if shorter < 2 {
            return [a.clone(), b.clone()];
        }

        let (i, j) = two_sorted_indices(rng, shorter + 1);
        let mut c1 = a.clone();
        let mut c2 = b.clone();
        c1[i..j].clone_from_slice(&b[i..j]);
        c2[i..j].clone_from_slice(&a[i..j]);

        [c1, c2]
    }
}

/// Uniform crossover: exchange each position of the common prefix with probability `swap`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformCrossover {
    /// The per-position exchange probability; `0.5` is customary.
    pub swap: f64,
}

impl Default for UniformCrossover {
    fn default() -> Self {
        Self { swap: 0.5 }
    }
}

impl<T: Clone> Crossover<Vec<T>> for UniformCrossover {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &Vec<T>, b: &Vec<T>, rng: &mut R) -> [Vec<T>; 2] {
        let mut c1 = a.clone();
        let mut c2 = b.clone();
        for k in 0..a.len().min(b.len()) {
            if unit_f64(rng) < self.swap {
                c1[k] = b[k].clone();
                c2[k] = a[k].clone();
            }
        }

        [c1, c2]
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::SplitMix64;

    /// Positional crossovers keep the per-position multiset of the parents.
    #[test]
    fn crossovers_exchange_positions() {
        let a = vec![0u8; 10];
        let b = vec![1u8; 10];
        let mut rng = SplitMix64::new(5);
        for _ in 0..50 {
            for [c1, c2] in [
                OnePoint.crossover(&a, &b, &mut rng),
                TwoPoint.crossover(&a, &b, &mut rng),
                UniformCrossover::default().crossover(&a, &b, &mut rng),
            ] {
                for k in 0..10 {
                    assert_eq!(c1[k] + c2[k], 1);
                }
            }
        }
    }

    /// One-point crossover of unequal parents swaps tails.
    #[test]
    fn one_point_unequal_lengths() {
        let [c1, c2] = OnePoint.crossover(&vec![0; 3], &vec![1; 7], &mut SplitMix64::new(2));

        assert_eq!(c1.len() + c2.len(), 10);
        assert!(c1.len() == 7 && c2.len() == 3);
    }
}
