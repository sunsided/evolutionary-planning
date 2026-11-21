//! Coordinated multi-robot planning at an intersection (after Parque 2024): RADES, DE and
//! CMA-ES search the lattice roadmap parameters of eight robots with the paper's budget of
//! 300 evaluations, and the best plan is checked for collisions.
//!
//! Run with `cargo run --release --example intersection`.

use evolutionary_planning::cmaes::{CmaEs, Config as CmaConfig};
use evolutionary_planning::de::rades::{Config as RadesConfig, Rades};
use evolutionary_planning::de::{Config as DeConfig, De};
use evolutionary_planning::problems::intersection::Intersection;
use evolutionary_planning::{Limits, Outcome, SplitMix64};

/// The seeds every optimiser runs with.
const SEEDS: core::ops::Range<u64> = 0..9;

/// The evaluation budget of every run: the paper's 300.
const EVALUATIONS: u64 = 300;

/// Runs the three optimisers over several seeds and prints the median and best costs.
fn main() {
    let mut scene = Intersection::crossing(8);
    let bounds = scene.bounds();
    println!(
        "8 robots, {EVALUATIONS} evaluations per run, {} seeds",
        SEEDS.end - SEEDS.start
    );

    report(&mut scene, "RADES", &mut |scene, seed| {
        Rades::new(RadesConfig::default(), bounds.clone()).run(
            scene,
            &mut Limits::evaluations(EVALUATIONS),
            &mut (),
            &mut SplitMix64::new(seed),
        )
    });
    report(&mut scene, "DE", &mut |scene, seed| {
        De::new(
            DeConfig {
                population_size: 10,
                ..DeConfig::default()
            },
            bounds.clone(),
        )
        .run(
            scene,
            &mut Limits::evaluations(EVALUATIONS),
            &mut (),
            &mut SplitMix64::new(seed),
        )
    });
    report(&mut scene, "CMA-ES", &mut |scene, seed| {
        CmaEs::new(
            bounds.center(),
            1.0,
            CmaConfig {
                bounds: Some(bounds.clone()),
                ..CmaConfig::default()
            },
        )
        .minimize(
            scene,
            &mut Limits::evaluations(EVALUATIONS),
            &mut (),
            &mut SplitMix64::new(seed),
        )
    });
}

/// Runs `optimise` once per seed and prints the median cost, the best cost, and the
/// collisions and makespan of the best plan.
fn report(
    scene: &mut Intersection,
    name: &str,
    optimise: &mut dyn FnMut(&mut Intersection, u64) -> Outcome<Vec<f64>>,
) {
    let mut costs = Vec::new();
    let mut best: Option<Outcome<Vec<f64>>> = None;
    for seed in SEEDS {
        let outcome = optimise(scene, seed);
        costs.push(outcome.best_cost());
        if best
            .as_ref()
            .is_none_or(|b| outcome.best_cost() < b.best_cost())
        {
            best = Some(outcome);
        }
    }

    costs.sort_by(f64::total_cmp);
    let best = best.expect("at least one seed");
    let plan = scene
        .plan(best.best_genome().expect("evaluated"))
        .expect("feasible");
    println!(
        "{name:>7}: median {:8.2}  best {:8.2}  collisions {}  makespan {}",
        costs[costs.len() / 2],
        best.best_cost(),
        scene.collisions(&plan),
        plan.iter().map(Vec::len).max().unwrap_or(0)
    );
}
