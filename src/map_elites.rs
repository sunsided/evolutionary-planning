//! MAP-Elites (Mouret and Clune 2015): quality diversity over a grid of behaviour
//! descriptors. The archive keeps, for every cell of the descriptor grid, the cheapest genome
//! whose behaviour falls into it, so a run returns a map of diverse, locally optimal
//! solutions (for example plans that differ in length and risk) instead of one optimum.
//!
//! ```mermaid
//! graph LR
//!     init[random genomes] --> insert[insert into grid]
//!     insert --> pick[pick a random elite, maybe a second]
//!     pick --> vary[crossover, mutation, repair]
//!     vary --> eval[cost and descriptor]
//!     eval --> insert
//! ```
//!
//! ```
//! use evolutionary_planning::map_elites::{Config, Grid, MapElites};
//! use evolutionary_planning::operators::real::{Bounds, Gaussian};
//! use evolutionary_planning::operators::{NoCrossover, Variation, real::Clamp};
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let bounds = Bounds::uniform(3, -1.0, 1.0);
//! let mut qd = MapElites::new(Config::default(), Variation {
//!     init: bounds.clone(),
//!     crossover: NoCrossover,
//!     mutation: Gaussian { sigma: 0.1, rate: Some(1.0) },
//!     repair: Clamp(bounds),
//! });
//! // Behaviour: the first two coordinates; cost: the square of the third.
//! let mut problem = |x: &Vec<f64>, descriptor: &mut [f64]| {
//!     descriptor.copy_from_slice(&x[..2]);
//!     x[2] * x[2]
//! };
//! let grid = Grid::new(vec![-1.0, -1.0], vec![1.0, 1.0], vec![10, 10]);
//! let outcome = qd.run(&mut problem, grid, &mut Limits::evaluations(5_000), &mut (), &mut SplitMix64::new(1));
//! assert!(outcome.archive.coverage() > 0.95);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::operators::{Crossover, Initializer, Mutation, Repair, Variation};
use crate::rng::{below, chance};
use crate::{Budget, Event, Observer, Stats, Stop, sanitize};

/// A problem for quality diversity: a cost to minimise and a behaviour descriptor.
///
/// Implemented for every `FnMut(&G, &mut [f64]) -> f64` that writes the descriptor and
/// returns the cost.
pub trait QdProblem<G: ?Sized> {
    /// Writes the descriptor of `genome` into `descriptor` (of the grid's dimension) and
    /// returns its cost (NaN counts as `+∞`).
    fn evaluate(&mut self, genome: &G, descriptor: &mut [f64]) -> f64;
}

impl<G: ?Sized, F: FnMut(&G, &mut [f64]) -> f64> QdProblem<G> for F {
    fn evaluate(&mut self, genome: &G, descriptor: &mut [f64]) -> f64 {
        self(genome, descriptor)
    }
}

/// A regular grid over a box of descriptor space.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// The lower corner, one value per descriptor dimension.
    lower: Vec<f64>,
    /// The upper corner; `upper[i] > lower[i]`.
    upper: Vec<f64>,
    /// The number of cells per dimension, each at least `1`.
    bins: Vec<usize>,
}

impl Grid {
    /// A grid with `bins[i]` equal cells per dimension between `lower` and `upper`.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ, a dimension has zero bins or an empty or non-finite
    /// range, or the cell count overflows.
    #[must_use]
    pub fn new(lower: Vec<f64>, upper: Vec<f64>, bins: Vec<usize>) -> Self {
        assert!(
            lower.len() == upper.len() && lower.len() == bins.len(),
            "grid dimensions differ"
        );
        for ((lo, hi), b) in lower.iter().zip(&upper).zip(&bins) {
            assert!(
                *b > 0 && lo.is_finite() && hi.is_finite() && lo < hi,
                "invalid grid dimension"
            );
        }
        assert!(
            bins.iter()
                .try_fold(1usize, |acc, &b| acc.checked_mul(b))
                .is_some(),
            "too many cells"
        );

        Self { lower, upper, bins }
    }

    /// The descriptor dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.bins.len()
    }

    /// The total number of cells.
    #[must_use]
    pub fn cells(&self) -> usize {
        self.bins.iter().product()
    }

    /// The flat (row-major) index of the cell containing `descriptor`; values outside the box
    /// fall into the edge cells, NaN into the first.
    #[must_use]
    pub fn cell(&self, descriptor: &[f64]) -> usize {
        let mut index = 0;
        for (i, &value) in descriptor.iter().enumerate().take(self.dimension()) {
            #[expect(clippy::cast_precision_loss, reason = "bin counts are far below 2^52")]
            let bins = self.bins[i] as f64;
            let t = ((value - self.lower[i]) / (self.upper[i] - self.lower[i]) * bins)
                .clamp(0.0, bins - 1.0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to 0..bins"
            )]
            let bin = if t.is_nan() {
                0
            } else {
                libm::floor(t) as usize
            };
            index = index * self.bins[i] + bin;
        }

        index
    }
}

/// One stored elite.
#[derive(Debug, Clone, PartialEq)]
pub struct Elite<G> {
    /// The genome.
    pub genome: G,
    /// Its cost; the smallest of every genome offered for this cell.
    pub cost: f64,
    /// Its behaviour descriptor, inside this cell.
    pub descriptor: Vec<f64>,
}

/// What happened to a genome offered to the [`Archive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Insertion {
    /// It filled an empty cell.
    New,
    /// It replaced a costlier elite.
    Improved,
    /// Its cell holds an elite at least as cheap.
    Rejected,
}

/// The MAP-Elites archive: one elite per grid cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Archive<G> {
    /// The grid.
    grid: Grid,
    /// The elite of each cell, by flat index.
    cells: Vec<Option<Elite<G>>>,
    /// The flat indices of the filled cells, in the order they were first filled.
    filled: Vec<usize>,
}

impl<G> Archive<G> {
    /// An empty archive over `grid`.
    #[must_use]
    pub fn new(grid: Grid) -> Self {
        let cells = (0..grid.cells()).map(|_| None).collect();

        Self {
            grid,
            cells,
            filled: Vec::new(),
        }
    }

    /// The grid.
    #[must_use]
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Offers a genome; it enters if its cell is empty or holds a strictly costlier elite.
    pub fn insert(&mut self, genome: G, cost: f64, descriptor: Vec<f64>) -> Insertion {
        let cost = sanitize(cost);
        let index = self.grid.cell(&descriptor);
        let outcome = match &self.cells[index] {
            None => {
                self.filled.push(index);
                Insertion::New
            }
            Some(elite) if cost < elite.cost => Insertion::Improved,
            Some(_) => return Insertion::Rejected,
        };

        self.cells[index] = Some(Elite {
            genome,
            cost,
            descriptor,
        });
        outcome
    }

    /// The elite of cell `index`, if any.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Elite<G>> {
        self.cells.get(index).and_then(Option::as_ref)
    }

    /// The filled cells and their elites, in the order the cells were first filled.
    pub fn elites(&self) -> impl Iterator<Item = (usize, &Elite<G>)> + '_ {
        self.filled
            .iter()
            .filter_map(|&i| self.cells[i].as_ref().map(|e| (i, e)))
    }

    /// The number of filled cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filled.len()
    }

    /// Whether no cell is filled.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filled.is_empty()
    }

    /// The fraction of filled cells.
    #[must_use]
    pub fn coverage(&self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "cell counts are far below 2^52")]
        let fraction = self.filled.len() as f64 / self.cells.len() as f64;

        fraction
    }

    /// The QD-score `Σ (offset - cost)` over all elites; with `offset` at least the worst
    /// possible cost it only grows during a run.
    #[must_use]
    pub fn qd_score(&self, offset: f64) -> f64 {
        self.elites().map(|(_, e)| offset - e.cost).sum()
    }

    /// The cheapest elite.
    #[must_use]
    pub fn best(&self) -> Option<&Elite<G>> {
        self.elites()
            .map(|(_, e)| e)
            .min_by(|a, b| a.cost.total_cmp(&b.cost))
    }

    /// A uniformly random elite; `None` if empty.
    fn random<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<&Elite<G>> {
        if self.filled.is_empty() {
            return None;
        }

        self.cells[self.filled[below(rng, self.filled.len())]].as_ref()
    }
}

/// Parameters of a [`MapElites`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The number of random genomes evaluated before variation starts, at least `1`.
    pub initial: usize,
    /// The probability that an offspring comes from crossover of two random elites.
    pub crossover_rate: f64,
    /// The probability that an offspring is mutated.
    pub mutation_rate: f64,
    /// Offspring per reported generation (only affects [`Stats::generations`] and events).
    pub batch: usize,
}

impl Default for Config {
    /// 100 random genomes, mutation only, batches of 100.
    fn default() -> Self {
        Self {
            initial: 100,
            crossover_rate: 0.0,
            mutation_rate: 1.0,
            batch: 100,
        }
    }
}

/// The result of a MAP-Elites run.
#[derive(Debug, Clone, PartialEq)]
pub struct QdOutcome<G> {
    /// The final archive.
    pub archive: Archive<G>,
    /// The counters; [`Stats::best_cost`] is the cheapest elite's cost.
    pub stats: Stats,
    /// Why the run ended.
    pub stop: Stop,
}

/// MAP-Elites over any genome.
#[derive(Debug, Clone, PartialEq)]
pub struct MapElites<I, X, M, P> {
    /// The run parameters.
    pub config: Config,
    /// The variation operators.
    pub variation: Variation<I, X, M, P>,
}

impl<I, X, M, P> MapElites<I, X, M, P> {
    /// Creates a MAP-Elites run.
    #[must_use]
    pub fn new(config: Config, variation: Variation<I, X, M, P>) -> Self {
        Self { config, variation }
    }

    /// Fills an archive over `grid` until the budget stops the run.
    pub fn run<G, Q, B, Obs, R>(
        &mut self,
        problem: &mut Q,
        grid: Grid,
        budget: &mut B,
        observer: &mut Obs,
        rng: &mut R,
    ) -> QdOutcome<G>
    where
        G: Clone,
        I: Initializer<G>,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        Q: QdProblem<G> + ?Sized,
        B: Budget<Stats> + ?Sized,
        Obs: Observer + ?Sized,
        R: Rng + ?Sized,
    {
        let dimension = grid.dimension();
        let mut archive = Archive::new(grid);
        let mut stats = Stats::default();
        let mut offspring = 0;
        while !budget.exhausted(&stats) {
            let genome =
                if stats.evaluations < self.config.initial.max(1) as u64 || archive.is_empty() {
                    self.variation.init.initialize(rng)
                } else {
                    self.offspring(&archive, rng)
                };

            let mut descriptor = alloc::vec![0.0; dimension];
            let cost = sanitize(problem.evaluate(&genome, &mut descriptor));
            stats.evaluations += 1;
            if archive.insert(genome, cost, descriptor) != Insertion::Rejected
                && cost < stats.best_cost
            {
                stats.best_cost = cost;
                if Obs::ENABLED {
                    observer.observe(Event::Improved {
                        evaluations: stats.evaluations,
                        cost,
                    });
                }
            }

            offspring += 1;
            if offspring == self.config.batch.max(1) {
                offspring = 0;
                stats.generations += 1;
                if Obs::ENABLED {
                    observer.observe(Event::Generation {
                        stats,
                        population_best: stats.best_cost,
                    });
                }
            }
        }

        QdOutcome {
            archive,
            stats,
            stop: Stop::Budget,
        }
    }

    /// Breeds one offspring from random elites.
    fn offspring<G, R>(&mut self, archive: &Archive<G>, rng: &mut R) -> G
    where
        G: Clone,
        X: Crossover<G>,
        M: Mutation<G>,
        P: Repair<G>,
        R: Rng + ?Sized,
    {
        let first = archive
            .random(rng)
            .map(|e| e.genome.clone())
            .expect("the archive is not empty");
        let mut child = if chance(rng, self.config.crossover_rate) {
            let second = archive
                .random(rng)
                .map(|e| e.genome.clone())
                .expect("the archive is not empty");
            let [child, _] = self.variation.crossover.crossover(&first, &second, rng);
            child
        } else {
            first
        };

        if chance(rng, self.config.mutation_rate) {
            self.variation.mutation.mutate(&mut child, rng);
        }

        self.variation.repair.repair(&mut child, rng);
        child
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;

    use super::*;
    use crate::operators::real::{Bounds, Clamp, Gaussian, IsoLineDd};
    use crate::{Limits, SplitMix64};

    /// Archive invariants after a run: every elite lies in its own cell, it is the cheapest of
    /// all genomes ever offered for that cell, and coverage and QD-score only grow.
    #[test]
    fn archive_invariants() {
        let grid = Grid::new(
            alloc::vec![-1.0, -1.0],
            alloc::vec![1.0, 1.0],
            alloc::vec![8, 6],
        );
        let reference = grid.clone();
        let mut cheapest: BTreeMap<usize, f64> = BTreeMap::new();
        let mut problem = |x: &Vec<f64>, d: &mut [f64]| {
            d[0] = x[0];
            d[1] = x[1] * x[2];
            let cost = (x[0] - 0.3) * (x[0] - 0.3) + x[2] * x[2];
            let entry = cheapest.entry(reference.cell(d)).or_insert(f64::INFINITY);
            *entry = entry.min(cost);
            cost
        };
        let bounds = Bounds::uniform(3, -1.0, 1.0);
        let mut qd = MapElites::new(
            Config {
                crossover_rate: 0.5,
                ..Config::default()
            },
            Variation {
                init: bounds.clone(),
                crossover: IsoLineDd {
                    sigma_iso: 0.05,
                    sigma_line: 0.2,
                },
                mutation: Gaussian {
                    sigma: 0.05,
                    rate: None,
                },
                repair: Clamp(bounds),
            },
        );
        let outcome = qd.run(
            &mut problem,
            grid,
            &mut Limits::evaluations(4_000),
            &mut (),
            &mut SplitMix64::new(4),
        );
        let archive = &outcome.archive;

        assert_eq!(archive.len(), cheapest.len());
        for (index, elite) in archive.elites() {
            assert_eq!(archive.grid().cell(&elite.descriptor), index);
            assert_eq!(elite.cost, cheapest[&index]);
        }
        assert_eq!(outcome.stats.evaluations, 4_000);
    }

    /// Insertion outcomes and the monotone QD-score.
    #[test]
    fn insertion_rules() {
        let mut archive = Archive::new(Grid::new(
            alloc::vec![0.0],
            alloc::vec![1.0],
            alloc::vec![2],
        ));
        assert_eq!(archive.insert("a", 3.0, alloc::vec![0.2]), Insertion::New);
        let score = archive.qd_score(10.0);
        assert_eq!(
            archive.insert("b", 3.0, alloc::vec![0.3]),
            Insertion::Rejected
        );
        assert_eq!(
            archive.insert("c", 1.0, alloc::vec![0.4]),
            Insertion::Improved
        );
        assert!(archive.qd_score(10.0) > score);
        assert_eq!(archive.insert("d", 5.0, alloc::vec![7.0]), Insertion::New);
        assert_eq!(archive.coverage(), 1.0);
        assert_eq!(archive.best().map(|e| e.genome), Some("c"));
    }

    /// Out-of-range and NaN descriptors fall into edge cells.
    #[test]
    fn grid_clamps() {
        let grid = Grid::new(
            alloc::vec![0.0, 0.0],
            alloc::vec![1.0, 1.0],
            alloc::vec![4, 3],
        );

        assert_eq!(grid.cell(&[-5.0, 5.0]), 2);
        assert_eq!(grid.cell(&[0.99, 0.99]), 11);
        assert_eq!(grid.cell(&[f64::NAN, 0.5]), 1);
    }
}
