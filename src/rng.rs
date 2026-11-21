//! The seeded pseudo-random generator [`SplitMix64`] and the sampling helpers every algorithm
//! uses: uniform floats, unbiased integers, standard normals and shuffles.
//!
//! An own generator rather than a dependency, so that a seed reproduces a run across releases
//! of this crate regardless of other crates' versions (see `docs/design.md`). It implements
//! `rand_core`'s generator trait, which only fixes the interface, not the sequence. Every
//! algorithm takes any `rand_core::Rng`, so callers can use another generator.

use core::convert::Infallible;

use rand_core::Rng;

/// The `SplitMix64` generator (Steele, Lea and Flood 2014).
///
/// Fast, with a full `2^64` period and good equidistribution for sampling. Not
/// cryptographically secure, which evolutionary search does not need.
///
/// ```
/// use evolutionary_planning::{SplitMix64, unit_f64};
///
/// let mut rng = SplitMix64::new(7);
/// assert!((0.0..1.0).contains(&unit_f64(&mut rng)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    /// The generator state, advanced by a fixed odd increment per draw.
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator from a seed. Every seed, including `0`, is valid.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64 random bits.
    fn next_word(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    /// Derives an independent generator for stream `stream`, for example one per restart or
    /// per decision, without consuming randomness from `self`.
    #[must_use]
    pub fn fork(&self, stream: u64) -> Self {
        let mixed = Self::new(self.state ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03)).next_word();

        Self::new(mixed)
    }
}

impl rand_core::TryRng for SplitMix64 {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok((self.next_word() >> 32) as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.next_word())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        rand_core::utils::fill_bytes_via_next_word(dst, || self.try_next_u64())
    }
}

/// Returns a uniform sample from `[0, 1)` with 53 random bits drawn from `rng`.
///
/// ```
/// use evolutionary_planning::{SplitMix64, unit_f64};
///
/// let mut rng = SplitMix64::new(1);
/// let u = unit_f64(&mut rng);
/// assert!(u >= 0.0 && u < 1.0);
/// ```
pub fn unit_f64<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    /// `2^-53`, the spacing of the 53-bit grid on `[0, 1)`.
    const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

    #[expect(
        clippy::cast_precision_loss,
        reason = "the value has at most 53 bits, which f64 represents exactly"
    )]
    let bits = (rng.next_u64() >> 11) as f64;

    bits * SCALE
}

/// Returns a uniform sample from `[low, high)`; `low` itself if the interval is empty.
pub fn uniform<R: Rng + ?Sized>(rng: &mut R, low: f64, high: f64) -> f64 {
    if high <= low {
        return low;
    }

    let x = low + (high - low) * unit_f64(rng);

    // Rounding can land exactly on `high`; keep the interval half-open.
    if x >= high { low } else { x }
}

/// Returns `true` with probability `p` (clamped to `[0, 1]`).
pub fn chance<R: Rng + ?Sized>(rng: &mut R, p: f64) -> bool {
    if p >= 1.0 {
        return true;
    }

    unit_f64(rng) < p
}

/// Returns a uniform integer from `0..n` with Lemire's unbiased multiply-and-reject method.
///
/// # Panics
///
/// Panics if `n == 0`.
///
/// ```
/// use evolutionary_planning::{SplitMix64, below};
///
/// let mut rng = SplitMix64::new(3);
/// assert!(below(&mut rng, 10) < 10);
/// ```
pub fn below<R: Rng + ?Sized>(rng: &mut R, n: usize) -> usize {
    assert!(n > 0, "below(0) has no valid result");

    let n = n as u64;
    let mut product = u128::from(rng.next_u64()) * u128::from(n);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the low half of a 128-bit product"
    )]
    let mut low = product as u64;
    if low < n {
        let threshold = n.wrapping_neg() % n;
        while low < threshold {
            product = u128::from(rng.next_u64()) * u128::from(n);

            #[expect(
                clippy::cast_possible_truncation,
                reason = "the low half of a 128-bit product"
            )]
            {
                low = product as u64;
            }
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the high half is below n, a usize"
    )]
    let index = (product >> 64) as usize;

    index
}

/// Returns a standard normal sample `N(0, 1)` with Marsaglia's polar method.
///
/// The second sample of each polar pair is discarded so the function needs no state; the
/// cost is one extra `ln` and `sqrt` per sample, small against any objective evaluation.
pub fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let u = 2.0 * unit_f64(rng) - 1.0;
        let v = 2.0 * unit_f64(rng) - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * libm::sqrt(-2.0 * libm::log(s) / s);
        }
    }
}

/// Shuffles `items` uniformly in place (Fisher-Yates).
pub fn shuffle<T, R: Rng + ?Sized>(rng: &mut R, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = below(rng, i + 1);
        items.swap(i, j);
    }
}

/// Returns two distinct uniform indices from `0..n`, the first strictly below the second.
///
/// # Panics
///
/// Panics if `n < 2`.
pub fn two_sorted_indices<R: Rng + ?Sized>(rng: &mut R, n: usize) -> (usize, usize) {
    assert!(n >= 2, "two distinct indices need n >= 2");

    let a = below(rng, n);
    let mut b = below(rng, n - 1);
    if b >= a {
        b += 1;
    }

    (a.min(b), a.max(b))
}

/// Returns `count` distinct indices from `0..n` that are all different from `exclude`, in
/// random order (partial Fisher-Yates over a scratch vector).
///
/// # Panics
///
/// Panics if fewer than `count` indices are available.
pub fn distinct_indices<R: Rng + ?Sized>(
    rng: &mut R,
    n: usize,
    count: usize,
    exclude: Option<usize>,
) -> alloc::vec::Vec<usize> {
    let mut pool: alloc::vec::Vec<usize> = (0..n).filter(|&i| Some(i) != exclude).collect();
    assert!(pool.len() >= count, "not enough distinct indices");

    for i in 0..count {
        let j = i + below(rng, pool.len() - i);
        pool.swap(i, j);
    }

    pool.truncate(count);
    pool
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first outputs for seed 0 match the reference implementation.
    #[test]
    fn matches_reference_sequence() {
        let mut rng = SplitMix64::new(0);

        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    /// `below` covers its range and never leaves it.
    #[test]
    fn below_covers_range() {
        let mut rng = SplitMix64::new(5);
        let mut seen = [false; 7];
        for _ in 0..1000 {
            seen[below(&mut rng, 7)] = true;
        }

        assert!(seen.iter().all(|&s| s));
    }

    /// The normal samples have roughly zero mean and unit variance.
    #[test]
    fn normal_moments() {
        let mut rng = SplitMix64::new(11);
        let n = 20_000;
        let samples: alloc::vec::Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / f64::from(n);
        let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / f64::from(n);

        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
    }

    /// Shuffling keeps the multiset, two indices are distinct and sorted.
    #[test]
    fn shuffle_and_pairs() {
        let mut rng = SplitMix64::new(2);
        let mut items: alloc::vec::Vec<u32> = (0..50).collect();
        shuffle(&mut rng, &mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();

        assert_eq!(sorted, (0..50).collect::<alloc::vec::Vec<_>>());
        for _ in 0..100 {
            let (a, b) = two_sorted_indices(&mut rng, 3);
            assert!(a < b && b < 3);
        }

        let picked = distinct_indices(&mut rng, 6, 5, Some(2));
        assert_eq!(picked.len(), 5);
        assert!(!picked.contains(&2));
    }

    /// Forked streams differ from each other and are reproducible.
    #[test]
    fn forks_are_distinct() {
        let base = SplitMix64::new(9);

        assert_ne!(base.fork(0), base.fork(1));
        assert_eq!(base.fork(3), base.fork(3));
    }
}
