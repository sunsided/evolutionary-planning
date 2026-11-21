//! Variation operators: the genome-agnostic traits ([`Initializer`], [`Mutation`],
//! [`Crossover`], [`Repair`], [`Selection`]), closure adapters, combinators, and ready-made
//! operators per representation in the submodules.
//!
//! Every trait method is generic over the generator, which keeps the hot path monomorphised;
//! closures are accepted through `*_fn` adapters that receive `&mut dyn Rng`.
//!
//! | Module | Genome |
//! |---|---|
//! | [`real`] | `Vec<f64>` within [`Bounds`](real::Bounds) |
//! | [`discrete`] | bit strings `Vec<bool>` and integer strings `Vec<u32>` |
//! | [`permutation`] | permutations `Vec<usize>` of `0..n` |
//! | [`sequence`] | any `Vec<T>`: fixed-length crossovers |
//! | [`chain`] | GLEAM action chains: variable-length `Vec<T>` and parameterised actions |
//! | [`selection`] | parent selection over a cost slice |

pub mod chain;
pub mod discrete;
pub mod permutation;
pub mod real;
pub mod selection;
pub mod sequence;

use alloc::vec::Vec;

use rand_core::Rng;

use crate::rng::{chance, unit_f64};

pub use selection::Selection;

/// Creates random genomes for an initial population.
pub trait Initializer<G> {
    /// Returns a new random genome.
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> G;
}

/// Changes one genome in place.
pub trait Mutation<G: ?Sized> {
    /// Mutates `genome`.
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R);
}

/// Produces two children from two parents.
pub trait Crossover<G> {
    /// Returns two children of `a` and `b`.
    fn crossover<R: Rng + ?Sized>(&mut self, a: &G, b: &G, rng: &mut R) -> [G; 2];
}

/// Makes a genome valid again after variation (genotypic repair).
///
/// Phenotypic repair, which only adjusts the interpretation, belongs in the objective.
pub trait Repair<G: ?Sized> {
    /// Repairs `genome` in place.
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R);
}

impl<G: ?Sized, M: Mutation<G> + ?Sized> Mutation<G> for &mut M {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R) {
        (**self).mutate(genome, rng);
    }
}

impl<G, X: Crossover<G> + ?Sized> Crossover<G> for &mut X {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &G, b: &G, rng: &mut R) -> [G; 2] {
        (**self).crossover(a, b, rng)
    }
}

impl<G, I: Initializer<G> + ?Sized> Initializer<G> for &mut I {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> G {
        (**self).initialize(rng)
    }
}

impl<G: ?Sized, P: Repair<G> + ?Sized> Repair<G> for &mut P {
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R) {
        (**self).repair(genome, rng);
    }
}

/// The repair that changes nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoRepair;

impl<G: ?Sized> Repair<G> for NoRepair {
    fn repair<R: Rng + ?Sized>(&mut self, _genome: &mut G, _rng: &mut R) {}
}

/// The mutation that changes nothing, for crossover-only or selection-only runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoMutation;

impl<G: ?Sized> Mutation<G> for NoMutation {
    fn mutate<R: Rng + ?Sized>(&mut self, _genome: &mut G, _rng: &mut R) {}
}

/// The crossover that returns copies of the parents, for mutation-only runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoCrossover;

impl<G: Clone> Crossover<G> for NoCrossover {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &G, b: &G, _rng: &mut R) -> [G; 2] {
        [a.clone(), b.clone()]
    }
}

/// Calls `f` with a generator behind `&mut dyn Rng`, for the closure adapters.
fn with_dyn<R: Rng + ?Sized, T>(rng: &mut R, f: impl FnOnce(&mut dyn Rng) -> T) -> T {
    let mut rng = rng;

    f(&mut rng)
}

/// Turns a closure `FnMut(&mut dyn Rng) -> G` into an [`Initializer`].
///
/// ```
/// use evolutionary_planning::operators::{Initializer, initializer_fn};
/// use evolutionary_planning::{SplitMix64, below};
///
/// let mut init = initializer_fn(|rng| below(rng, 10));
/// assert!(init.initialize(&mut SplitMix64::new(1)) < 10);
/// ```
pub fn initializer_fn<G, F: FnMut(&mut dyn Rng) -> G>(f: F) -> InitializerFn<F> {
    InitializerFn(f)
}

/// A closure used as an [`Initializer`]; created by [`initializer_fn`].
#[derive(Debug, Clone, Copy)]
pub struct InitializerFn<F>(F);

impl<G, F: FnMut(&mut dyn Rng) -> G> Initializer<G> for InitializerFn<F> {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> G {
        with_dyn(rng, |rng| (self.0)(rng))
    }
}

/// Turns a closure `FnMut(&mut G, &mut dyn Rng)` into a [`Mutation`].
pub fn mutation_fn<G: ?Sized, F: FnMut(&mut G, &mut dyn Rng)>(f: F) -> MutationFn<F> {
    MutationFn(f)
}

/// A closure used as a [`Mutation`]; created by [`mutation_fn`].
#[derive(Debug, Clone, Copy)]
pub struct MutationFn<F>(F);

impl<G: ?Sized, F: FnMut(&mut G, &mut dyn Rng)> Mutation<G> for MutationFn<F> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R) {
        with_dyn(rng, |rng| (self.0)(genome, rng));
    }
}

/// Turns a closure `FnMut(&G, &G, &mut dyn Rng) -> [G; 2]` into a [`Crossover`].
pub fn crossover_fn<G, F: FnMut(&G, &G, &mut dyn Rng) -> [G; 2]>(f: F) -> CrossoverFn<F> {
    CrossoverFn(f)
}

/// A closure used as a [`Crossover`]; created by [`crossover_fn`].
#[derive(Debug, Clone, Copy)]
pub struct CrossoverFn<F>(F);

impl<G, F: FnMut(&G, &G, &mut dyn Rng) -> [G; 2]> Crossover<G> for CrossoverFn<F> {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &G, b: &G, rng: &mut R) -> [G; 2] {
        with_dyn(rng, |rng| (self.0)(a, b, rng))
    }
}

/// Turns a closure `FnMut(&mut G, &mut dyn Rng)` into a [`Repair`].
pub fn repair_fn<G: ?Sized, F: FnMut(&mut G, &mut dyn Rng)>(f: F) -> RepairFn<F> {
    RepairFn(f)
}

/// A closure used as a [`Repair`]; created by [`repair_fn`].
#[derive(Debug, Clone, Copy)]
pub struct RepairFn<F>(F);

impl<G: ?Sized, F: FnMut(&mut G, &mut dyn Rng)> Repair<G> for RepairFn<F> {
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R) {
        with_dyn(rng, |rng| (self.0)(genome, rng));
    }
}

/// Applies `a` with probability `weight_a / (weight_a + weight_b)`, otherwise `b`; the
/// building block of GLEAM-style operator mixes. Nest it for more than two operators, or
/// use [`mix`].
///
/// ```
/// use evolutionary_planning::operators::{mix, permutation::{Inversion, Swap}};
///
/// // Inversions twice as often as swaps.
/// let _mutation = mix(Inversion, 2.0, Swap, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mix<A, B> {
    /// The first operator.
    pub a: A,
    /// The second operator.
    pub b: B,
    /// The probability of choosing `a`, in `[0, 1]`.
    pub probability_a: f64,
}

/// Creates a [`Mix`] choosing `a` and `b` in proportion to their (non-negative) weights.
///
/// # Panics
///
/// Panics if a weight is negative or both are zero.
#[must_use]
pub fn mix<A, B>(a: A, weight_a: f64, b: B, weight_b: f64) -> Mix<A, B> {
    assert!(
        weight_a >= 0.0 && weight_b >= 0.0 && weight_a + weight_b > 0.0,
        "mix weights must be non-negative and not both zero"
    );

    Mix {
        a,
        b,
        probability_a: weight_a / (weight_a + weight_b),
    }
}

impl<G: ?Sized, A: Mutation<G>, B: Mutation<G>> Mutation<G> for Mix<A, B> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R) {
        if unit_f64(rng) < self.probability_a {
            self.a.mutate(genome, rng);
        } else {
            self.b.mutate(genome, rng);
        }
    }
}

impl<G, A: Crossover<G>, B: Crossover<G>> Crossover<G> for Mix<A, B> {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &G, b: &G, rng: &mut R) -> [G; 2] {
        if unit_f64(rng) < self.probability_a {
            self.a.crossover(a, b, rng)
        } else {
            self.b.crossover(a, b, rng)
        }
    }
}

/// Applies `first`, then `second`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Then<A, B> {
    /// Applied first.
    pub first: A,
    /// Applied to the result of `first`.
    pub second: B,
}

impl<G: ?Sized, A: Mutation<G>, B: Mutation<G>> Mutation<G> for Then<A, B> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R) {
        self.first.mutate(genome, rng);
        self.second.mutate(genome, rng);
    }
}

impl<G: ?Sized, A: Repair<G>, B: Repair<G>> Repair<G> for Then<A, B> {
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R) {
        self.first.repair(genome, rng);
        self.second.repair(genome, rng);
    }
}

/// Applies `inner` with probability `probability`, otherwise leaves the genome unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sometimes<M> {
    /// The wrapped mutation.
    pub inner: M,
    /// The probability of applying it, in `[0, 1]`.
    pub probability: f64,
}

impl<G: ?Sized, M: Mutation<G>> Mutation<G> for Sometimes<M> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut G, rng: &mut R) {
        if chance(rng, self.probability) {
            self.inner.mutate(genome, rng);
        }
    }
}

/// An initialiser that yields caller-supplied seed genomes first (for example previous plans
/// or heuristic solutions), then falls back to `inner`.
///
/// Jakob (2021) advises seeding at most about 20 % of the initial population so it still
/// covers the search space; [`Seeded::new`] keeps at most `max_fraction` of
/// `population_size` seeds.
#[derive(Debug, Clone, PartialEq)]
pub struct Seeded<G, I> {
    /// The remaining seeds, yielded from the back (stored reversed).
    seeds: Vec<G>,
    /// The initialiser used once the seeds run out.
    inner: I,
}

impl<G, I> Seeded<G, I> {
    /// Keeps the first `floor(max_fraction * population_size)` (at least one, if any) of
    /// `seeds`, in order.
    #[must_use]
    pub fn new(mut seeds: Vec<G>, inner: I, population_size: usize, max_fraction: f64) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "a population fraction, clamped to [0, population_size]"
        )]
        let cap = ((max_fraction.clamp(0.0, 1.0) * population_size as f64) as usize).max(1);
        seeds.truncate(cap);
        seeds.reverse();

        Self { seeds, inner }
    }
}

impl<G, I: Initializer<G>> Initializer<G> for Seeded<G, I> {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> G {
        match self.seeds.pop() {
            Some(seed) => seed,
            None => self.inner.initialize(rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::SplitMix64;

    /// A mix with weight zero on one side always picks the other.
    #[test]
    fn mix_respects_weights() {
        let mut rng = SplitMix64::new(1);
        let mut m = mix(
            mutation_fn(|x: &mut u32, _| *x += 1),
            1.0,
            mutation_fn(|x: &mut u32, _| *x += 100),
            0.0,
        );
        let mut value = 0;
        for _ in 0..10 {
            m.mutate(&mut value, &mut rng);
        }

        assert_eq!(value, 10);
    }

    /// Seeds come first and are capped.
    #[test]
    fn seeded_caps_and_orders() {
        let mut rng = SplitMix64::new(1);
        let mut init = Seeded::new(vec![1, 2, 3], initializer_fn(|_| 0), 10, 0.2);
        let got: Vec<i32> = (0..4).map(|_| init.initialize(&mut rng)).collect();

        assert_eq!(got, [1, 2, 0, 0]);
    }
}
