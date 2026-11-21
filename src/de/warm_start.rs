//! A library of solved problems used to warm-start new searches, after the pre-motion
//! library of Yue et al. (2023): solutions are stored under a task descriptor (for example a
//! target state), and a new task whose descriptor lies within a radius of a stored one starts
//! from that solution instead of from scratch.

use alloc::vec::Vec;

/// Stored solutions keyed by task descriptors, looked up by Euclidean distance.
///
/// A linear scan: libraries of hundreds of entries (Yue et al. store 567) are scanned in
/// microseconds, and an index would only pay off for far larger ones.
///
/// ```
/// use evolutionary_planning::de::WarmStartLibrary;
///
/// let mut library = WarmStartLibrary::new();
/// library.insert(vec![0.6, 0.2], vec![1.0, 2.0, 3.0]);
/// library.insert(vec![0.0, 0.0], vec![0.0, 0.0, 0.0]);
///
/// let (distance, solution) = library.nearest(&[0.58, 0.21], 0.05).expect("within radius");
/// assert!(distance < 0.05);
/// assert_eq!(solution, &[1.0, 2.0, 3.0]);
/// assert!(library.nearest(&[0.3, 0.3], 0.05).is_none());
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WarmStartLibrary {
    /// `(descriptor, solution)` pairs in insertion order.
    entries: Vec<(Vec<f64>, Vec<f64>)>,
}

impl WarmStartLibrary {
    /// An empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `solution` for the task described by `descriptor`.
    pub fn insert(&mut self, descriptor: Vec<f64>, solution: Vec<f64>) {
        self.entries.push((descriptor, solution));
    }

    /// The number of stored solutions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the library is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The solution whose descriptor is nearest to `query`, with its distance, if that
    /// distance is at most `radius`. Ties go to the earlier entry. Descriptors of a different
    /// dimension never match.
    #[must_use]
    pub fn nearest(&self, query: &[f64], radius: f64) -> Option<(f64, &[f64])> {
        self.entries
            .iter()
            .filter(|(descriptor, _)| descriptor.len() == query.len())
            .map(|(descriptor, solution)| (distance(descriptor, query), solution.as_slice()))
            .filter(|(d, _)| *d <= radius)
            .min_by(|a, b| a.0.total_cmp(&b.0))
    }

    /// Up to `k` solutions within `radius` of `query`, nearest first: seeds for a population.
    #[must_use]
    pub fn neighbours(&self, query: &[f64], radius: f64, k: usize) -> Vec<Vec<f64>> {
        let mut hits: Vec<(f64, usize)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, (descriptor, _))| descriptor.len() == query.len())
            .map(|(index, (descriptor, _))| (distance(descriptor, query), index))
            .filter(|(d, _)| *d <= radius)
            .collect();
        hits.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        hits.into_iter()
            .take(k)
            .map(|(_, index)| self.entries[index].1.clone())
            .collect()
    }
}

/// The Euclidean distance of two equally long vectors.
fn distance(a: &[f64], b: &[f64]) -> f64 {
    libm::sqrt(a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::de::{Config, De};
    use crate::operators::real::Bounds;
    use crate::{Limits, SplitMix64};

    /// A warm start from a nearby solved task reaches the target with fewer evaluations.
    #[test]
    fn warm_start_saves_evaluations() {
        let solve = |target: [f64; 4], seeds: Vec<Vec<f64>>| {
            let mut de = De::new(Config::default(), Bounds::uniform(4, -10.0, 10.0));
            let mut objective = |x: &Vec<f64>| {
                x.iter()
                    .zip(&target)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
            };
            de.run_from(
                seeds,
                &mut objective,
                &mut Limits::evaluations(50_000).with_target(1e-8),
                &mut (),
                &mut SplitMix64::new(1),
            )
        };

        let first = solve([3.0, -2.0, 5.0, 1.0], Vec::new());
        let mut library = WarmStartLibrary::new();
        library.insert(
            alloc::vec![3.0, -2.0],
            first.best_genome().expect("solved").clone(),
        );

        let seeds = library.neighbours(&[3.05, -2.0], 0.1, 4);
        let warm = solve([3.05, -2.0, 5.0, 1.0], seeds);
        let cold = solve([3.05, -2.0, 5.0, 1.0], Vec::new());

        assert!(warm.best_cost() <= 1e-8 && cold.best_cost() <= 1e-8);
        assert!(
            warm.stats.evaluations < cold.stats.evaluations,
            "{} vs {}",
            warm.stats.evaluations,
            cold.stats.evaluations
        );
    }
}
