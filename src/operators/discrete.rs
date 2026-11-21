//! Operators for bit strings `Vec<bool>` and integer strings `Vec<u32>` with per-gene
//! domains: random initialisation, bit flips, random reset and creep mutation. Crossovers
//! for both live in [`sequence`](super::sequence).

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Initializer, Mutation, Repair};
use crate::rng::{below, chance};

/// Uniform random bit strings of a fixed length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomBits {
    /// The number of bits.
    pub len: usize,
}

impl Initializer<Vec<bool>> for RandomBits {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<bool> {
        (0..self.len).map(|_| rng.next_u32() & 1 == 1).collect()
    }
}

/// Flips each bit with probability `rate` (default `1/n`).
///
/// ```
/// use evolutionary_planning::operators::{Mutation, discrete::BitFlip};
/// use evolutionary_planning::SplitMix64;
///
/// let mut bits = vec![false; 8];
/// BitFlip { rate: Some(1.0) }.mutate(&mut bits, &mut SplitMix64::new(1));
/// assert!(bits.iter().all(|&b| b));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BitFlip {
    /// The per-bit probability; `None` means `1/n`.
    pub rate: Option<f64>,
}

impl Mutation<Vec<bool>> for BitFlip {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<bool>, rng: &mut R) {
        #[expect(clippy::cast_precision_loss, reason = "lengths are far below 2^52")]
        let rate = self.rate.unwrap_or(1.0 / genome.len().max(1) as f64);
        for bit in genome.iter_mut() {
            if chance(rng, rate) {
                *bit = !*bit;
            }
        }
    }
}

/// Per-gene domains `0..size[i]` of an integer string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Domains(pub Vec<u32>);

impl Domains {
    /// The same domain `0..size` for `len` genes.
    #[must_use]
    pub fn uniform(len: usize, size: u32) -> Self {
        Self(alloc::vec![size; len])
    }
}

/// Uniform random integer strings.
///
/// # Panics
///
/// Panics on an empty domain (`size == 0`).
impl Initializer<Vec<u32>> for Domains {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<u32> {
        self.0.iter().map(|&size| draw(rng, size)).collect()
    }
}

/// Draws uniformly from `0..size`.
fn draw<R: Rng + ?Sized>(rng: &mut R, size: u32) -> u32 {
    u32::try_from(below(rng, size as usize)).expect("below(size) < size, a u32")
}

/// Replaces each gene with probability `rate` (default `1/n`) by a uniform value of its
/// domain.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomReset {
    /// The per-gene domains.
    pub domains: Domains,
    /// The per-gene probability; `None` means `1/n`.
    pub rate: Option<f64>,
}

impl Mutation<Vec<u32>> for RandomReset {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<u32>, rng: &mut R) {
        #[expect(clippy::cast_precision_loss, reason = "lengths are far below 2^52")]
        let rate = self.rate.unwrap_or(1.0 / genome.len().max(1) as f64);
        for (gene, &size) in genome.iter_mut().zip(&self.domains.0) {
            if chance(rng, rate) {
                *gene = draw(rng, size);
            }
        }
    }
}

/// Creep mutation: adds a uniform step in `-max_step..=max_step` (non-zero) to each gene with
/// probability `rate` (default `1/n`), clamped to the domain. Respects causality for ordinal
/// integers, where random reset does not.
#[derive(Debug, Clone, PartialEq)]
pub struct Creep {
    /// The per-gene domains.
    pub domains: Domains,
    /// The largest step, at least `1`.
    pub max_step: u32,
    /// The per-gene probability; `None` means `1/n`.
    pub rate: Option<f64>,
}

impl Mutation<Vec<u32>> for Creep {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<u32>, rng: &mut R) {
        #[expect(clippy::cast_precision_loss, reason = "lengths are far below 2^52")]
        let rate = self.rate.unwrap_or(1.0 / genome.len().max(1) as f64);
        let max_step = self.max_step.max(1);
        for (gene, &size) in genome.iter_mut().zip(&self.domains.0) {
            if size <= 1 || !chance(rng, rate) {
                continue;
            }

            let step = 1 + draw(rng, max_step);
            *gene = if rng.next_u32() & 1 == 0 {
                gene.saturating_sub(step)
            } else {
                gene.saturating_add(step).min(size - 1)
            };
        }
    }
}

/// Clamps every gene into its domain.
impl Repair<Vec<u32>> for Domains {
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut Vec<u32>, _rng: &mut R) {
        for (gene, &size) in genome.iter_mut().zip(&self.0) {
            *gene = (*gene).min(size.saturating_sub(1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMix64;

    /// Integer operators stay within their domains.
    #[test]
    fn integer_operators_respect_domains() {
        let mut domains = Domains(alloc::vec![1, 2, 5, 100]);
        let mut rng = SplitMix64::new(3);
        let mut reset = RandomReset {
            domains: domains.clone(),
            rate: Some(1.0),
        };
        let mut creep = Creep {
            domains: domains.clone(),
            max_step: 3,
            rate: Some(1.0),
        };
        for _ in 0..200 {
            let mut g = domains.initialize(&mut rng);
            reset.mutate(&mut g, &mut rng);
            creep.mutate(&mut g, &mut rng);
            assert!(g.iter().zip(&domains.0).all(|(v, s)| v < s));
        }
    }
}
