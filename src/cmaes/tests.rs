//! Tests of CMA-ES against the known convergence behaviour on standard functions.

use alloc::vec::Vec;

use super::*;
use crate::problems::single::{ellipsoid, rosenbrock, sphere};
use crate::{Limits, SplitMix64};

/// Runs CMA-ES from `start` with σ = `sigma` and returns the outcome.
fn minimize(
    f: fn(&[f64]) -> f64,
    start: Vec<f64>,
    sigma: f64,
    evaluations: u64,
    seed: u64,
) -> (Outcome<Vec<f64>>, CmaEs) {
    let mut cma = CmaEs::new(start, sigma, Config::default());
    let mut objective = |x: &Vec<f64>| f(x);
    let outcome = cma.minimize(
        &mut objective,
        &mut Limits::evaluations(evaluations).with_target(1e-10),
        &mut (),
        &mut SplitMix64::new(seed),
    );

    (outcome, cma)
}

/// The sphere in 10D reaches 1e-10 in well under 2500 evaluations (the tutorial's Figure 4
/// shows roughly 1500).
#[test]
fn sphere_converges_fast() {
    let (outcome, _) = minimize(sphere, alloc::vec![1.0; 10], 0.5, 2_500, 1);

    assert!(outcome.best_cost() <= 1e-10, "{}", outcome.best_cost());
}

/// On the ellipsoid with condition 1e6, C learns the inverse Hessian: its condition number
/// approaches 1e6.
#[test]
fn ellipsoid_learns_condition() {
    let (outcome, cma) = minimize(ellipsoid, alloc::vec![1.0; 10], 0.5, 15_000, 2);
    let condition = cma.condition();

    assert!(outcome.best_cost() <= 1e-10, "{}", outcome.best_cost());
    assert!((1e5..1e7).contains(&condition), "condition {condition}");
}

/// Rosenbrock in 6D: most seeds reach the global optimum; the rest end in the known local
/// minimum near (-1, 1, ..., 1) (cost about 3.7), never anywhere else.
#[test]
fn rosenbrock_global_or_known_local() {
    let mut global = 0;
    for seed in 0..6 {
        let (outcome, _) = minimize(rosenbrock, alloc::vec![0.0; 6], 0.5, 40_000, seed);
        if outcome.best_cost() <= 1e-10 {
            global += 1;
        } else {
            assert!(
                outcome.best_cost() < 4.0,
                "seed {seed}: {}",
                outcome.best_cost()
            );
        }
    }

    assert!(global >= 4, "{global} of 6 runs found the global optimum");
}

/// CMA-ES is rank-based: a strictly increasing transform of the objective yields the same
/// samples and means, bit for bit.
#[test]
fn invariant_under_monotone_transform() {
    let mut a = CmaEs::new(alloc::vec![0.5; 5], 0.3, Config::default());
    let mut b = a.clone();
    let mut rng_a = SplitMix64::new(7);
    let mut rng_b = SplitMix64::new(7);
    for _ in 0..60 {
        let costs_a: Vec<f64> = a.ask(&mut rng_a).iter().map(|x| rosenbrock(x)).collect();
        let costs_b: Vec<f64> = b
            .ask(&mut rng_b)
            .iter()
            .map(|x| libm::exp(rosenbrock(x)) * 3.0 - 7.0)
            .collect();
        a.tell(&costs_a);
        b.tell(&costs_b);
    }

    assert_eq!(a.mean(), b.mean());
    assert_eq!(a.sigma(), b.sigma());
}

/// Without a budget the run ends through a B.3 criterion; on the sphere TolFun or TolX.
#[test]
fn terminates_by_criteria() {
    let mut cma = CmaEs::new(alloc::vec![1.0; 4], 1.0, Config::default());
    let mut objective = |x: &Vec<f64>| sphere(x);
    let outcome = cma.minimize(
        &mut objective,
        &mut Limits::unlimited(),
        &mut (),
        &mut SplitMix64::new(3),
    );

    assert!(
        matches!(
            outcome.stop,
            Stop::CmaEs(Termination::TolFun | Termination::TolX)
        ),
        "{:?}",
        outcome.stop
    );
    assert!(outcome.best_cost() < 1e-12);
}

/// A flat objective ends with EqualFunValues.
#[test]
fn flat_objective_stops() {
    let mut cma = CmaEs::new(
        alloc::vec![0.0; 3],
        1.0,
        Config {
            tol_fun: 0.0,
            ..Config::default()
        },
    );
    let mut objective = |_: &Vec<f64>| 1.0;
    let outcome = cma.minimize(
        &mut objective,
        &mut Limits::evaluations(100_000),
        &mut (),
        &mut SplitMix64::new(3),
    );

    assert_eq!(outcome.stop, Stop::CmaEs(Termination::EqualFunValues));
}

/// A far too small initial step size on a linear function triggers TolXUp.
#[test]
fn tiny_sigma_on_linear_function() {
    let mut cma = CmaEs::new(alloc::vec![0.0; 3], 1e-8, Config::default());
    let mut objective = |x: &Vec<f64>| x.iter().sum::<f64>();
    let outcome = cma.minimize(
        &mut objective,
        &mut Limits::evaluations(100_000),
        &mut (),
        &mut SplitMix64::new(3),
    );

    assert_eq!(outcome.stop, Stop::CmaEs(Termination::TolXUp));
}

/// The bound penalty keeps the reported best inside the box and finds the corner optimum.
#[test]
fn box_bounds_by_penalty() {
    let config = Config {
        bounds: Some(Bounds::uniform(3, -1.0, 1.0)),
        ..Config::default()
    };
    let mut cma = CmaEs::new(alloc::vec![0.0; 3], 0.5, config);
    let mut objective = |x: &Vec<f64>| x.iter().map(|v| (v - 2.0) * (v - 2.0)).sum::<f64>();
    let outcome = cma.minimize(
        &mut objective,
        &mut Limits::evaluations(5_000),
        &mut (),
        &mut SplitMix64::new(4),
    );
    let best = outcome.best_genome().expect("evaluated");

    assert!(best.iter().all(|v| (-1.0..=1.0).contains(v)));
    assert!(
        (outcome.best_cost() - 3.0).abs() < 1e-6,
        "{}",
        outcome.best_cost()
    );
}

/// Same seed, same run; exact evaluation budget.
#[test]
fn deterministic_and_exact_budget() {
    let run = || minimize(sphere, alloc::vec![1.0; 5], 0.5, 333, 9).0;

    assert_eq!(run(), run());
    assert_eq!(run().stats.evaluations, 333);
}

/// IPOP restarts with growing λ and reports them.
#[test]
fn ipop_restarts() {
    let mut restarts = Vec::new();
    let outcome = {
        let mut observer = crate::observer_fn(|e| {
            if let Event::Restart { population, .. } = e {
                restarts.push(population);
            }
        });
        let mut ipop = Ipop::new(Config::default(), Bounds::uniform(5, -5.12, 5.12));
        let mut objective = |x: &Vec<f64>| crate::problems::single::rastrigin(x);
        ipop.run(
            &mut objective,
            &mut Limits::evaluations(300_000).with_target(1e-8),
            &mut observer,
            &mut SplitMix64::new(5),
        )
    };

    assert!(outcome.best_cost() <= 1e-8, "{}", outcome.best_cost());
    assert!(restarts.windows(2).all(|w| w[1] > w[0]));
}
