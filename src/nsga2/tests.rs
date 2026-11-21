//! Tests of NSGA-II against brute-force Pareto fronts on small discrete problems.

use alloc::vec::Vec;

use super::*;
use crate::operators::discrete::{BitFlip, RandomBits};
use crate::operators::sequence::UniformCrossover;
use crate::operators::{NoRepair, sequence::OnePoint};
use crate::pareto::non_dominated;
use crate::problems::multi::{Knapsack, Lotz, enumerate_bits};
use crate::{Limits, SplitMix64};

/// The distinct objective vectors of the brute-force feasible Pareto front.
pub(crate) fn brute_front<M: MultiObjective<Vec<bool>>>(
    problem: &mut M,
    n: usize,
) -> Vec<Vec<f64>> {
    let feasible: Vec<Vec<f64>> = enumerate_bits(problem, n)
        .into_iter()
        .filter(|(_, _, v)| *v <= 0.0)
        .map(|(_, o, _)| o)
        .collect();
    let mut front: Vec<Vec<f64>> = non_dominated(&feasible)
        .into_iter()
        .map(|i| feasible[i].clone())
        .collect();
    front.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    front.dedup();

    front
}

/// The distinct objective vectors of an outcome's front.
pub(crate) fn distinct<G>(front: &[Evaluated<G>]) -> Vec<Vec<f64>> {
    let mut points: Vec<Vec<f64>> = front.iter().map(|e| e.objectives.clone()).collect();
    points.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    points.dedup();

    points
}

/// An NSGA-II over bit strings.
fn bits_nsga2(
    len: usize,
    population_size: usize,
) -> Nsga2<RandomBits, UniformCrossover, BitFlip, NoRepair> {
    Nsga2::new(
        Config {
            population_size,
            ..Config::default()
        },
        Variation {
            init: RandomBits { len },
            crossover: UniformCrossover::default(),
            mutation: BitFlip::default(),
            repair: NoRepair,
        },
    )
}

/// On a constrained bi-objective knapsack, every returned point is feasible and on the
/// brute-force front, and the whole front is found.
#[test]
fn knapsack_front_is_exact() {
    let mut problem = Knapsack::generate(12, 7);
    let expected = brute_front(&mut problem, 12);
    let outcome = bits_nsga2(12, 100).run(
        &mut problem,
        &mut Limits::evaluations(30_000),
        &mut (),
        &mut SplitMix64::new(1),
    );
    let found = distinct(&outcome.front);

    assert!(outcome.front.iter().all(Evaluated::is_feasible));
    assert!(
        found.iter().all(|p| expected.contains(p)),
        "{found:?} vs {expected:?}"
    );
    assert_eq!(found, expected);
}

/// LOTZ: the n + 1 front points are all found with one-point crossover.
#[test]
fn lotz_front_is_covered() {
    let n = 10;
    let mut nsga2 = Nsga2::new(
        Config {
            population_size: 40,
            ..Config::default()
        },
        Variation {
            init: RandomBits { len: n },
            crossover: OnePoint,
            mutation: BitFlip::default(),
            repair: NoRepair,
        },
    );
    let outcome = nsga2.run(
        &mut Lotz,
        &mut Limits::evaluations(40_000),
        &mut (),
        &mut SplitMix64::new(2),
    );

    assert_eq!(distinct(&outcome.front), brute_front(&mut Lotz, n));
}

/// Same seed, same run; exact budget; population size kept.
#[test]
fn deterministic() {
    let run = || {
        bits_nsga2(8, 20).run(
            &mut Lotz,
            &mut Limits::evaluations(777),
            &mut (),
            &mut SplitMix64::new(3),
        )
    };
    let outcome = run();

    assert_eq!(outcome, run());
    assert_eq!(outcome.stats.evaluations, 777);
    assert_eq!(outcome.population.len(), 20);
}
