//! An evolutionary programmer after Lou, Ding and Wu (2022): a meta-GA evolves bit strings
//! that encode complete path-planning GAs (operators and their parameters, drawn from an
//! operator library), scored by how well the decoded planner solves a set of training
//! instances within a fixed evaluation budget.
//!
//! The genome has 18 bits (the paper's has 64 over a larger library):
//!
//! | Bits | Gene | Values |
//! |---|---|---|
//! | 0–1 | selection operator | tournament, truncation, roulette, linear ranking with SUS |
//! | 2–3 | selection parameter | size 2/3/4/6; fraction 0.2/0.3/0.5/0.7; pressure 1.2/1.4/1.7/2 |
//! | 4–5 | elitism | 0, 1, 2, 4 |
//! | 6–7 | crossover rate | 0, 0.4, 0.7, 0.95 |
//! | 8–9 | mutation rate | 0.2, 0.5, 0.8, 1 |
//! | 10–12 | population size | 10, 16, 24, 32, 48, 64, 96, 128 |
//! | 13–14 | reroute expansions | 8, 32, 128, 512 |
//! | 15 | scheme | generational, steady state |
//! | 16–17 | steady-state replacement | always (codes 0, 1), only if better (codes 2, 3) |
//!
//! The meta-GA uses linear ranking with selective pressure `1.2`, the paper's scores from
//! 0.8 (worst) to 1.2 (best).
//!
//! ```
//! use evolutionary_planning::graph_path::Grid;
//! use evolutionary_planning::programmer::{Instance, MetaConfig, Programmer};
//! use evolutionary_planning::SplitMix64;
//!
//! let mut grid = Grid::open(8, 8, false);
//! for y in 0..6 {
//!     grid.block(4, y);
//! }
//! let instances = vec![Instance::new(&grid, grid.node(0, 0), grid.node(7, 0))];
//! let mut programmer = Programmer::new(instances, 400);
//! let (planner, score) = programmer.evolve(&MetaConfig { population_size: 10, evaluations: 60 }, &mut SplitMix64::new(1));
//! assert!(score.is_finite() && planner.population_size >= 10);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::ga::{Config as GaConfig, Ga, Operators, Scheme};
use crate::graph_path::{
    CommonVertex, Graph, PathLength, RandomPath, RemoveLoops, Reroute, dijkstra,
};
use crate::operators::discrete::{BitFlip, RandomBits};
use crate::operators::selection::{
    LinearRanking, Proportional, Sampling, Selection, Tournament, Truncation,
};
use crate::operators::sequence::UniformCrossover;
use crate::{Limits, Objective, SplitMix64};

/// The number of bits of a planner genome.
pub const GENOME_BITS: usize = 18;

/// A selection operator chosen by the genome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionChoice {
    /// A tournament of the given size.
    Tournament(usize),
    /// Truncation to the given fraction.
    Truncation(f64),
    /// Fitness-proportional roulette.
    Roulette,
    /// Linear ranking with the given pressure, sampled universally.
    Ranking(f64),
}

impl Selection for SelectionChoice {
    fn select<R: Rng + ?Sized>(
        &mut self,
        costs: &[f64],
        count: usize,
        rng: &mut R,
        out: &mut Vec<usize>,
    ) {
        match *self {
            Self::Tournament(size) => Tournament { size }.select(costs, count, rng, out),
            Self::Truncation(fraction) => Truncation { fraction }.select(costs, count, rng, out),
            Self::Roulette => Proportional {
                sampling: Sampling::Roulette,
            }
            .select(costs, count, rng, out),
            Self::Ranking(pressure) => LinearRanking {
                pressure,
                sampling: Sampling::Universal,
            }
            .select(costs, count, rng, out),
        }
    }
}

/// A decoded path-planning GA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannerConfig {
    /// The parent selection.
    pub selection: SelectionChoice,
    /// Elites kept per generation.
    pub elitism: usize,
    /// The crossover probability.
    pub crossover_rate: f64,
    /// The mutation probability.
    pub mutation_rate: f64,
    /// The population size.
    pub population_size: usize,
    /// The largest detour search of the reroute mutation.
    pub max_expansions: usize,
    /// Generational or steady state.
    pub scheme: Scheme,
}

impl PlannerConfig {
    /// Decodes a [`GENOME_BITS`]-bit genome (missing bits read as `false`).
    #[must_use]
    pub fn decode(bits: &[bool]) -> Self {
        let field = |start: usize, len: usize| {
            (start..start + len).fold(0usize, |acc, i| {
                acc << 1 | usize::from(bits.get(i).copied().unwrap_or(false))
            })
        };
        let parameter = field(2, 2);
        let selection = match field(0, 2) {
            0 => SelectionChoice::Tournament([2, 3, 4, 6][parameter]),
            1 => SelectionChoice::Truncation([0.2, 0.3, 0.5, 0.7][parameter]),
            2 => SelectionChoice::Roulette,
            _ => SelectionChoice::Ranking([1.2, 1.4, 1.7, 2.0][parameter]),
        };
        let scheme = if field(15, 1) == 0 {
            Scheme::Generational
        } else {
            Scheme::SteadyState {
                only_if_better: field(16, 2) >= 2,
            }
        };

        Self {
            selection,
            elitism: [0, 1, 2, 4][field(4, 2)],
            crossover_rate: [0.0, 0.4, 0.7, 0.95][field(6, 2)],
            mutation_rate: [0.2, 0.5, 0.8, 1.0][field(8, 2)],
            population_size: [10, 16, 24, 32, 48, 64, 96, 128][field(10, 3)],
            max_expansions: [8, 32, 128, 512][field(13, 2)],
            scheme,
        }
    }

    /// Runs the decoded planner on one instance; returns the best path length found.
    pub fn solve<G: Graph + ?Sized, R: Rng + ?Sized>(
        &self,
        graph: &G,
        start: usize,
        goal: usize,
        evaluations: u64,
        rng: &mut R,
    ) -> f64 {
        let mut ga = Ga::new(
            GaConfig {
                population_size: self.population_size,
                elitism: self.elitism,
                crossover_rate: self.crossover_rate,
                mutation_rate: self.mutation_rate,
                scheme: self.scheme,
            },
            Operators {
                init: RandomPath { graph, start, goal },
                selection: self.selection,
                crossover: CommonVertex,
                mutation: Reroute {
                    graph,
                    max_expansions: self.max_expansions,
                },
                repair: RemoveLoops,
            },
        );
        let mut objective = PathLength { graph, start, goal };

        ga.run(
            &mut objective,
            &mut Limits::evaluations(evaluations),
            &mut (),
            rng,
        )
        .best_cost()
    }
}

/// A training instance: a graph, a start, a goal and the optimal length (from Dijkstra).
#[derive(Debug, Clone, Copy)]
pub struct Instance<'g, G: ?Sized> {
    /// The graph.
    pub graph: &'g G,
    /// The start vertex.
    pub start: usize,
    /// The goal vertex.
    pub goal: usize,
    /// The shortest path length, `+∞` if unreachable.
    pub optimum: f64,
}

impl<'g, G: Graph + ?Sized> Instance<'g, G> {
    /// An instance with its optimum computed by Dijkstra.
    #[must_use]
    pub fn new(graph: &'g G, start: usize, goal: usize) -> Self {
        let optimum = dijkstra(graph, start, goal).map_or(f64::INFINITY, |(_, c)| c);

        Self {
            graph,
            start,
            goal,
            optimum,
        }
    }
}

/// Parameters of the meta-GA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetaConfig {
    /// The number of planner genomes per generation.
    pub population_size: usize,
    /// The number of planner genomes evaluated in total.
    pub evaluations: u64,
}

/// The planner-programming objective: the mean relative excess `cost / optimum - 1` of the
/// decoded planner over the training instances (a failed run counts `10`), each run with a
/// fixed seed so a genome's score is deterministic.
#[derive(Debug, Clone)]
pub struct Programmer<'g, G: ?Sized> {
    /// The training instances (reachable ones only contribute).
    pub instances: Vec<Instance<'g, G>>,
    /// The evaluation budget of every planner run.
    pub evaluations: u64,
}

impl<'g, G: Graph + ?Sized> Programmer<'g, G> {
    /// A programmer for `instances` with a per-run budget.
    #[must_use]
    pub fn new(instances: Vec<Instance<'g, G>>, evaluations: u64) -> Self {
        Self {
            instances,
            evaluations,
        }
    }

    /// The score of a decoded planner (smaller is better).
    #[must_use]
    pub fn score(&self, planner: &PlannerConfig) -> f64 {
        let reachable: Vec<&Instance<'g, G>> = self
            .instances
            .iter()
            .filter(|i| i.optimum.is_finite())
            .collect();
        let total: f64 = reachable
            .iter()
            .enumerate()
            .map(|(k, instance)| {
                let mut rng = SplitMix64::new(0x5EED ^ k as u64);
                let cost = planner.solve(
                    instance.graph,
                    instance.start,
                    instance.goal,
                    self.evaluations,
                    &mut rng,
                );
                if cost.is_finite() {
                    cost / instance.optimum.max(f64::MIN_POSITIVE) - 1.0
                } else {
                    10.0
                }
            })
            .sum();

        #[expect(clippy::cast_precision_loss, reason = "instance counts are small")]
        let mean = total / reachable.len().max(1) as f64;

        mean
    }

    /// Runs the meta-GA and returns the best planner with its score.
    pub fn evolve<R: Rng + ?Sized>(
        &mut self,
        meta: &MetaConfig,
        rng: &mut R,
    ) -> (PlannerConfig, f64) {
        let mut ga = Ga::new(
            GaConfig {
                population_size: meta.population_size.max(2),
                elitism: 1,
                crossover_rate: 0.8,
                mutation_rate: 1.0,
                scheme: Scheme::Generational,
            },
            Operators {
                init: RandomBits { len: GENOME_BITS },
                selection: LinearRanking {
                    pressure: 1.2,
                    sampling: Sampling::Roulette,
                },
                crossover: UniformCrossover::default(),
                mutation: BitFlip::default(),
                repair: crate::operators::NoRepair,
            },
        );
        let outcome = ga.run(
            self,
            &mut Limits::evaluations(meta.evaluations),
            &mut (),
            rng,
        );
        let bits = outcome
            .best_genome()
            .cloned()
            .unwrap_or_else(|| alloc::vec![false; GENOME_BITS]);

        (PlannerConfig::decode(&bits), outcome.best_cost())
    }
}

impl<G: Graph + ?Sized> Objective<Vec<bool>> for Programmer<'_, G> {
    fn evaluate(&mut self, genome: &Vec<bool>) -> f64 {
        self.score(&PlannerConfig::decode(genome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_path::Grid;
    use crate::operators::Initializer;

    /// Three mazes as training instances.
    fn mazes() -> Vec<Grid> {
        (0..3)
            .map(|k| {
                let mut grid = Grid::open(14, 10, k == 2);
                for y in 0..8 {
                    grid.block(3 + k, y);
                }
                for y in 2..10 {
                    grid.block(8 + k, y);
                }
                grid
            })
            .collect()
    }

    /// Every bit string decodes to a valid planner, and decoding is a pure function.
    #[test]
    fn decoding_is_total() {
        let mut rng = SplitMix64::new(1);
        let mut init = RandomBits { len: GENOME_BITS };
        for _ in 0..200 {
            let bits = init.initialize(&mut rng);
            let config = PlannerConfig::decode(&bits);
            assert_eq!(config, PlannerConfig::decode(&bits));
            assert!(config.population_size >= 10 && config.elitism <= 4);
        }
    }

    /// The evolved planner scores at least as well as the median of 40 random planners.
    #[test]
    fn evolved_planner_beats_random_ones() {
        let grids = mazes();
        let instances: Vec<Instance<'_, Grid>> = grids
            .iter()
            .map(|g| Instance::new(g, g.node(0, 0), g.node(13, 9)))
            .collect();
        let mut programmer = Programmer::new(instances, 300);
        let (_, evolved) = programmer.evolve(
            &MetaConfig {
                population_size: 12,
                evaluations: 120,
            },
            &mut SplitMix64::new(2),
        );

        let mut rng = SplitMix64::new(7);
        let mut init = RandomBits { len: GENOME_BITS };
        let mut random: Vec<f64> = (0..40)
            .map(|_| programmer.evaluate(&init.initialize(&mut rng)))
            .collect();
        random.sort_by(f64::total_cmp);

        assert!(
            evolved <= random[20],
            "evolved {evolved}, random median {}",
            random[20]
        );
    }
}
