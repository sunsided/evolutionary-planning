//! Tests of the generational and steady-state GA.

use alloc::vec::Vec;

use super::*;
use crate::operators::discrete::{BitFlip, RandomBits};
use crate::operators::permutation::{Inversion, Ox, RandomPermutation};
use crate::operators::real::{Bounds, Polynomial, Sbx};
use crate::operators::selection::{LinearRanking, Tournament};
use crate::operators::sequence::UniformCrossover;
use crate::operators::{NoRepair, mix};
use crate::problems::single::{Tour, one_max, sphere};
use crate::{Limits, SplitMix64};

/// A GA on OneMax with the given scheme.
fn one_max_ga(scheme: Scheme) -> Ga<RandomBits, Tournament, UniformCrossover, BitFlip, NoRepair> {
    Ga::new(
        Config {
            population_size: 40,
            scheme,
            ..Config::default()
        },
        Operators {
            init: RandomBits { len: 40 },
            selection: Tournament { size: 2 },
            crossover: UniformCrossover::default(),
            mutation: BitFlip::default(),
            repair: NoRepair,
        },
    )
}

/// Both schemes solve OneMax.
#[test]
fn solves_one_max() {
    for scheme in [
        Scheme::Generational,
        Scheme::SteadyState {
            only_if_better: true,
        },
    ] {
        let mut objective = |b: &Vec<bool>| one_max(b);
        let outcome = one_max_ga(scheme).run(
            &mut objective,
            &mut Limits::evaluations(40_000).with_target(0.0),
            &mut (),
            &mut SplitMix64::new(3),
        );

        assert_eq!(outcome.best_cost(), 0.0, "{scheme:?}");
    }
}

/// The evaluation limit is exact, and the same seed gives the same run.
#[test]
fn exact_budget_and_determinism() {
    let run = |seed| {
        let mut objective = |b: &Vec<bool>| one_max(b);
        one_max_ga(Scheme::Generational).run(
            &mut objective,
            &mut Limits::evaluations(1_234),
            &mut (),
            &mut SplitMix64::new(seed),
        )
    };
    let a = run(5);

    assert_eq!(a.stats.evaluations, 1_234);
    assert_eq!(a, run(5));
    assert_eq!(a.population.len(), 40);
    assert!(a.population.windows(2).all(|w| w[0].cost <= w[1].cost));
}

/// With elitism, the population's best never gets worse from generation to generation.
#[test]
fn elitism_is_monotone() {
    let mut history = Vec::new();
    let mut observer = crate::observer_fn(|event| {
        if let crate::Event::Generation {
            population_best, ..
        } = event
        {
            history.push(population_best);
        }
    });
    let mut objective = |b: &Vec<bool>| one_max(b);
    one_max_ga(Scheme::Generational).run(
        &mut objective,
        &mut Limits::generations(30),
        &mut observer,
        &mut SplitMix64::new(8),
    );

    assert_eq!(history.len(), 30);
    assert!(history.windows(2).all(|w| w[1] <= w[0]), "{history:?}");
}

/// A real-coded GA gets close to the sphere optimum.
#[test]
fn real_coded_sphere() {
    let bounds = Bounds::uniform(5, -5.0, 5.0);
    let mut ga = Ga::new(
        Config {
            elitism: 2,
            ..Config::default()
        },
        Operators {
            init: bounds.clone(),
            selection: LinearRanking::default(),
            crossover: Sbx {
                bounds: bounds.clone(),
                eta: 15.0,
            },
            mutation: Polynomial {
                bounds,
                eta: 20.0,
                rate: None,
            },
            repair: NoRepair,
        },
    );
    let mut objective = |x: &Vec<f64>| sphere(x);
    let outcome = ga.run(
        &mut objective,
        &mut Limits::evaluations(30_000),
        &mut (),
        &mut SplitMix64::new(2),
    );

    assert!(outcome.best_cost() < 1e-3, "{}", outcome.best_cost());
}

/// A permutation GA finds the optimal tour of a small circle instance.
#[test]
fn permutation_tour() {
    let tour = Tour::circle(12);
    let mut ga = Ga::new(
        Config {
            population_size: 60,
            elitism: 2,
            ..Config::default()
        },
        Operators {
            init: RandomPermutation { n: 12 },
            selection: Tournament { size: 3 },
            crossover: Ox,
            mutation: mix(Inversion, 3.0, crate::operators::permutation::Swap, 1.0),
            repair: NoRepair,
        },
    );
    let mut objective = |p: &Vec<usize>| tour.length(p);
    let outcome = ga.run(
        &mut objective,
        &mut Limits::evaluations(60_000),
        &mut (),
        &mut SplitMix64::new(4),
    );

    assert!(
        (outcome.best_cost() - Tour::circle_optimum(12)).abs() < 1e-9,
        "{}",
        outcome.best_cost()
    );
}

/// A budget that allows nothing returns no best and an empty population.
#[test]
fn zero_budget() {
    let mut objective = |b: &Vec<bool>| one_max(b);
    let outcome = one_max_ga(Scheme::Generational).run(
        &mut objective,
        &mut Limits::evaluations(0),
        &mut (),
        &mut SplitMix64::new(1),
    );

    assert!(outcome.best.is_none() && outcome.population.is_empty());
}

/// Elitism of the whole population is clamped to `size - 1`, so every generation still breeds
/// (and the budget still stops the run).
#[test]
fn full_elitism_still_breeds() {
    let mut ga = one_max_ga(Scheme::Generational);
    ga.config.elitism = 1_000;
    let mut objective = |b: &Vec<bool>| one_max(b);
    let outcome = ga.run(
        &mut objective,
        &mut Limits::evaluations(500),
        &mut (),
        &mut SplitMix64::new(1),
    );

    assert_eq!(outcome.stats.evaluations, 500);
}
