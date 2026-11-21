//! GLEAM action chains end to end: HyGLEAM's diffusion model evolves a variable-length
//! program of typed, parameterised actions ("move a distance", "turn by an integer angle")
//! that drives a robot to a target.

use evolutionary_planning::diffusion::{Config, Diffusion};
use evolutionary_planning::ga::Operators;
use evolutionary_planning::operators::chain::{
    Action, ActionKind, ActionModel, CutAndSplice, DeleteGene, InsertGene, ParamNewValue,
    ParamShift, ParamSpec, RandomChain, SegmentInversion, SegmentShift,
};
use evolutionary_planning::operators::selection::LinearRanking;
use evolutionary_planning::operators::{Mutation, NoRepair, mix};
use evolutionary_planning::{Limits, SplitMix64};

/// Executes a chain from the origin heading east: kind 0 moves `params[0]` forward, kind 1
/// turns by `params[0]` degrees. Returns the final position.
fn execute(chain: &[Action]) -> (f64, f64) {
    let (mut x, mut y, mut heading) = (0.0f64, 0.0f64, 0.0f64);
    for action in chain {
        match action.kind {
            0 => {
                x += action.params[0] * heading.to_radians().cos();
                y += action.params[0] * heading.to_radians().sin();
            }
            _ => heading += action.params[0],
        }
    }

    (x, y)
}

/// GLEAM's mix of gene-level (parameter shift and new value) and chain-level (insertion,
/// deletion, segment shift and inversion) mutations.
fn chain_mutation(model: &ActionModel) -> impl Mutation<Vec<Action>> {
    mix(
        mix(
            ParamShift {
                model: model.clone(),
            },
            4.0,
            ParamNewValue {
                model: model.clone(),
            },
            1.0,
        ),
        3.0,
        mix(
            mix(
                InsertGene {
                    genes: model.clone(),
                    max_len: 12,
                },
                1.0,
                DeleteGene { min_len: 1 },
                1.0,
            ),
            1.0,
            mix(
                SegmentShift {
                    max_segment: Some(3),
                },
                1.0,
                SegmentInversion {
                    max_segment: Some(3),
                },
                1.0,
            ),
            1.0,
        ),
        1.0,
    )
}

/// The robot reaches within 0.1 of the target (3, 4) with a short program.
#[test]
fn chain_program_reaches_target() {
    let model = ActionModel::new(vec![
        ActionKind::new(vec![ParamSpec::real(0.0, 3.0)]),
        ActionKind::new(vec![ParamSpec::integer(-90.0, 90.0)]),
    ]);
    let mutation = chain_mutation(&model);
    let mut diffusion = Diffusion::new(
        Config::default(),
        Operators {
            init: RandomChain {
                genes: model.clone(),
                min_len: 2,
                max_len: 8,
            },
            selection: LinearRanking::default(),
            crossover: CutAndSplice { max_len: 12 },
            mutation,
            repair: NoRepair,
        },
    );
    let mut objective = |chain: &Vec<Action>| {
        let (x, y) = execute(chain);
        (x - 3.0).hypot(y - 4.0)
            + 0.001 * f64::from(u32::try_from(chain.len()).expect("short chains"))
    };
    let outcome = diffusion.run(
        &mut objective,
        &mut Limits::evaluations(30_000),
        &mut (),
        &mut SplitMix64::new(1),
    );
    let best = outcome.best_genome().expect("evaluated");
    let (x, y) = execute(best);

    assert!(
        (x - 3.0).hypot(y - 4.0) < 0.1,
        "reached ({x}, {y}) with {} actions",
        best.len()
    );
    assert!(best.iter().all(|a| model.is_valid(a)));
}
