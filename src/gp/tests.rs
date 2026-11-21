//! Tests of the GP operators and an evolved navigation policy.

use alloc::vec::Vec;

use proptest::prelude::*;

use super::*;
use crate::ga::{Config, Ga, Operators, Scheme};
use crate::graph_path::Grid;
use crate::operators::selection::Tournament;
use crate::operators::{Mix, mix};
use crate::{Limits, SplitMix64};

/// The navigation language: symbols 0–3 are `if blocked(N/E/S/W) then a else b`, 4 and 5 are
/// `if goal is east / north then a else b`, 6–9 are the moves N, E, S, W.
fn navigation() -> Grammar {
    Grammar::new(alloc::vec![2, 2, 2, 2, 2, 2, 0, 0, 0, 0])
}

/// The move offsets of N, E, S, W.
const MOVES: [(isize, isize); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

/// Whether `(x, y)` is a free cell of `grid`; coordinates may be negative.
fn free(grid: &Grid, x: isize, y: isize) -> bool {
    matches!((usize::try_from(x), usize::try_from(y)), (Ok(x), Ok(y)) if grid.is_free(x, y))
}

/// The move the policy picks at `(x, y)` heading for `goal`.
fn decide(
    grammar: &Grammar,
    program: &[Symbol],
    at: usize,
    grid: &Grid,
    pos: (isize, isize),
    goal: (isize, isize),
) -> usize {
    let kids = children(grammar, program, at);
    let symbol = usize::from(program[at]);
    let branch = |condition: bool| if condition { kids[0] } else { kids[1] };
    match symbol {
        0..=3 => {
            let (dx, dy) = MOVES[symbol];
            let next = branch(!free(grid, pos.0 + dx, pos.1 + dy));
            decide(grammar, program, next, grid, pos, goal)
        }
        4 => decide(grammar, program, branch(goal.0 > pos.0), grid, pos, goal),
        5 => decide(grammar, program, branch(goal.1 > pos.1), grid, pos, goal),
        _ => symbol - 6,
    }
}

/// Runs the policy for 60 steps; returns the final Manhattan distance to the goal.
fn run_policy(
    grammar: &Grammar,
    program: &[Symbol],
    grid: &Grid,
    start: (isize, isize),
    goal: (isize, isize),
) -> usize {
    let mut pos = start;
    for _ in 0..60 {
        if pos == goal {
            return 0;
        }

        let (dx, dy) = MOVES[decide(grammar, program, 0, grid, pos, goal)];
        if free(grid, pos.0 + dx, pos.1 + dy) {
            pos = (pos.0 + dx, pos.1 + dy);
        }
    }

    (goal.0 - pos.0).unsigned_abs() + (goal.1 - pos.1).unsigned_abs()
}

/// Three training worlds in which heading straight for the goal gets stuck: a wall across
/// the bottom-left approach, a wall across the top-right approach, and both.
fn worlds() -> Vec<Grid> {
    (0..3)
        .map(|k| {
            let mut grid = Grid::open(10, 10, false);
            if k != 1 {
                for y in 0..8 {
                    grid.block(4, y);
                }
            }
            if k != 0 {
                for x in 2..10 {
                    grid.block(x, 6);
                }
            }
            grid
        })
        .collect()
}

proptest! {
    /// Initialisation and every operator produce valid trees within the limits.
    #[test]
    fn operators_keep_trees_valid(seed in any::<u64>()) {
        let grammar = navigation();
        let mut rng = SplitMix64::new(seed);
        let mut init = RampedHalfAndHalf { grammar: grammar.clone(), min_depth: 1, max_depth: 5 };
        let mut crossover = SubtreeCrossover { grammar: grammar.clone(), max_depth: 7, max_len: 60 };
        let mut mutation = mix(
            SubtreeMutation { grammar: grammar.clone(), subtree_depth: 3, max_depth: 7, max_len: 60 },
            1.0,
            PointMutation { grammar: grammar.clone() },
            1.0,
        );
        let a = init.initialize(&mut rng);
        let b = init.initialize(&mut rng);
        prop_assert!(is_valid(&grammar, &a) && depth(&grammar, &a) <= 5);
        for mut child in crossover.crossover(&a, &b, &mut rng) {
            mutation.mutate(&mut child, &mut rng);
            prop_assert!(is_valid(&grammar, &child));
            prop_assert!(child.len() <= 60.max(a.len()).max(b.len()) && depth(&grammar, &child) <= 7.max(depth(&grammar, &a)).max(depth(&grammar, &b)));
        }
    }
}

/// Tree helpers on a hand-built program: `if-east(E, if-blocked-N(W, N))`.
#[test]
fn tree_helpers() {
    let grammar = navigation();
    let program = [4, 7, 0, 9, 6];

    assert!(is_valid(&grammar, &program));
    assert!(!is_valid(&grammar, &program[..4]));
    assert_eq!(subtree_end(&grammar, &program, 2), 5);
    assert_eq!(children(&grammar, &program, 0), [1, 2]);
    assert_eq!(depth(&grammar, &program), 2);
}

/// The GP setup of the navigation test: 200 programs, tournaments of four, subtree crossover,
/// subtree and point mutation, depth at most 8 and at most 80 nodes.
fn navigation_ga(
    grammar: &Grammar,
) -> Ga<
    RampedHalfAndHalf,
    Tournament,
    SubtreeCrossover,
    Mix<SubtreeMutation, PointMutation>,
    crate::operators::NoRepair,
> {
    Ga::new(
        Config {
            population_size: 200,
            elitism: 2,
            crossover_rate: 0.9,
            mutation_rate: 0.3,
            scheme: Scheme::Generational,
        },
        Operators {
            init: RampedHalfAndHalf {
                grammar: grammar.clone(),
                min_depth: 2,
                max_depth: 5,
            },
            selection: Tournament { size: 4 },
            crossover: SubtreeCrossover {
                grammar: grammar.clone(),
                max_depth: 8,
                max_len: 80,
            },
            mutation: mix(
                SubtreeMutation {
                    grammar: grammar.clone(),
                    subtree_depth: 3,
                    max_depth: 8,
                    max_len: 80,
                },
                1.0,
                PointMutation {
                    grammar: grammar.clone(),
                },
                1.0,
            ),
            repair: crate::operators::NoRepair,
        },
    )
}

/// GP evolves a reactive policy over three worlds and three starts that is at least as good
/// as the best simple hand-written one ("north unless blocked, then east") and leaves less
/// than a quarter of the greedy policy's remaining distance. (Memoryless policies cannot
/// solve every episode: some need to go back and forth along a wall.)
#[test]
fn evolves_navigation_policy() {
    let grammar = navigation();
    let grids = worlds();
    let mut objective = |program: &Vec<Symbol>| {
        let total: usize = grids
            .iter()
            .flat_map(|g| {
                [(0, 0), (9, 0), (0, 4)]
                    .map(|start| run_policy(&grammar, program, g, start, (9, 9)))
            })
            .sum();
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let cost = total as f64 + 0.001 * program.len() as f64;
        cost
    };
    let mut ga = navigation_ga(&grammar);
    let outcome = ga.run(
        &mut objective,
        &mut Limits::evaluations(20_000),
        &mut (),
        &mut SplitMix64::new(1),
    );

    let greedy = alloc::vec![4, 7, 6];

    let greedy_cost = objective(&greedy);
    let north_then_east = alloc::vec![0, 7, 6];
    let north_cost = objective(&north_then_east);

    assert!(
        greedy_cost >= 1.0 && north_cost >= 1.0,
        "the worlds defeat simple policies"
    );
    assert!(
        outcome.best_cost() <= north_cost && outcome.best_cost() * 4.0 < greedy_cost,
        "evolved {}",
        outcome.best_cost()
    );
}
