//! Evolutionary search under a temporal-logic constraint (the EvoPlan use case without the
//! neural parts): NSGA-II trades control effort against progress for a point mass, subject
//! to an STL speed limit used as a graded constraint violation.

use evolutionary_planning::nsga2::{Config, Nsga2};
use evolutionary_planning::operators::real::{Bounds, Polynomial, Sbx};
use evolutionary_planning::operators::{NoRepair, Variation};
use evolutionary_planning::stl::Formula;
use evolutionary_planning::{Limits, SplitMix64, multi_objective_fn};

/// The trace `[speed, position]` of a 1D point mass under accelerations `controls`.
fn trace(controls: &[f64]) -> Vec<[f64; 2]> {
    let (mut x, mut v) = (0.0, 0.0);
    controls
        .iter()
        .map(|a| {
            v += 0.1 * a;
            x += 0.1 * v;
            [v, x]
        })
        .collect()
}

/// Every member of the returned front satisfies "always, speed at most 0.5" and "eventually
/// position at least 0.3", and the front is not trivial.
#[test]
fn front_satisfies_the_stl_constraint() {
    let steps = 20;
    let phi = Formula::and(vec![
        Formula::always(0, steps, Formula::le(0, 0.5)),
        Formula::eventually(0, steps, Formula::ge(1, 0.3)),
    ]);
    let mut problem = multi_objective_fn(2, |u: &Vec<f64>, out: &mut [f64]| {
        let t = trace(u);
        out[0] = u.iter().map(|a| a * a).sum();
        out[1] = -t[t.len() - 1][1];
        phi.violation(&t)
    });
    let bounds = Bounds::uniform(steps, -2.0, 2.0);
    let mut nsga2 = Nsga2::new(
        Config::default(),
        Variation {
            init: bounds.clone(),
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
    let outcome = nsga2.run(
        &mut problem,
        &mut Limits::evaluations(30_000),
        &mut (),
        &mut SplitMix64::new(1),
    );

    assert!(outcome.front.len() > 5);
    for member in &outcome.front {
        assert!(member.is_feasible());
        assert!(phi.robustness(&trace(&member.genome)) >= 0.0);
    }
}
